import os

from typing import Tuple, Sequence, Callable
from copy import deepcopy
import pickle
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import NearestNeighbors

from robot_action_execution.predicates.predicate_utils import MetaPredicateData, PredicateLibraryBase
from robot_action_execution.action_execution_model_utils import FailureSearchParams

class ActionExecutionModel(object):
    # action name
    name = None

    # object class to which the model is applied (class taken from
    # an ontology if model generalisation to other classes is desired)
    object_type = None

    # action preconditions
    preconditions = None

    # Gaussian process regressor
    gpr = None

    # generated execution samples
    sample_buffer = None

    # min/max limits of the action parameters
    # (stored as a list of tuples, one entry per parameter)
    parameter_limits = None

    # parameters used to search for action parameterisations
    # that make the action fail (search is performed along a
    # Gaussian distribution with increasing standard deviations)
    failure_search_params = None

    # a dictionary mapping parameter names (as specified in the
    # predicate's 'relation_parameter_causes') to parameter indices
    # as represented by the execution model's Gaussian process
    parameter_idx_mappings = None

    def __init__(self, name: str, object_type: str,
                 model_file_path: str=None):
        '''
        Keyword arguments:
        name: str -- name of the action
        object_type: str -- object type for which the action is applicable
        model_file_path: str -- path of a saved model (default None)

        '''
        self.name = name
        self.object_type = object_type
        self.sample_buffer = []
        if model_file_path:
            self.load(model_file_path)
        else:
            print('Initialising empty model')
            self.preconditions = dict()
            self.gpr = None

    def learn_preconditions(self, **kwargs):
        raise NotImplementedError('learn_predicates needs to be overriden')

    def learn_success_model(self, **kwargs):
        raise NotImplementedError('learn_success_model needs to be overriden')

    def clear_sample_buffer(self):
        '''Self-explanatory title - clears the sample buffer,
        thereby refreshing the sample generation.
        '''
        self.sample_buffer = []

    def sample_state(self, value: int,
                     predicate_library: PredicateLibraryBase,
                     state_update_fn: Callable,
                     mode=None, use_sample_buffer=False, **kwargs):
        '''Returns a state for which the underlying GP equals "value".

        Keyword arguments:
        value: int -- function value for which an state should be sampled
        predicate_library: PredicateLibraryBase -- predicate library class used for defining the
                                                   preconditions of the action
        state_update_fn: Callable -- function that predictively applies samples parameters
                                     so that the preconditions can be verified
        mode: str -- qualitative mode under which the action is executed (default None)
        use_sample_buffer: bool -- whether to use a sample buffer, which prevents
                                   previous samples from being reused - useful
                                   for backtracking search (default False)

        '''
        state = None
        precondition_values = None
        if mode is not None:
            precondition_values = [p[2] for p in self.preconditions[mode]]
        else:
            precondition_values = [p[2] for p in self.preconditions]

        sample_found = False
        execution_data = np.array(self.gpr.X_train_)
        value_indices = np.where(self.gpr.y_train_ == value)[0]
        constraints = kwargs.get('constraints', None)

        if constraints:
            non_constrained_param_idx = [i for i, cx in enumerate(constraints)
                                         if cx is None]
            constrained_training_data = np.array(self.gpr.X_train_[value_indices])
            constrained_training_data[:, non_constrained_param_idx] = 0.

            n_neighbours = int(self.gpr.X_train_.shape[0] * 0.1)
            n_neighbours = n_neighbours if n_neighbours < constrained_training_data.shape[0] \
                                        else constrained_training_data.shape[0]
            neighbour_tree = NearestNeighbors(n_neighbors=n_neighbours,
                                              algorithm='ball_tree').fit(constrained_training_data)

            constrained_instance = np.array([c if c is not None else 0.
                                             for c in constraints])[np.newaxis]
            _, indices = neighbour_tree.kneighbors(constrained_instance)
            execution_data = np.array(self.gpr.X_train_[value_indices])
            value_indices = indices[0]

        while not sample_found:
            # we take a random input from the training set
            # where the function is equal to "value"
            random_state_idx = np.random.choice(value_indices)
            state = np.array(execution_data[random_state_idx])

            # the returned state is sampled from a normal
            # distribution centered at the selected input
            # and with a standard deviation equal to
            # the GP standard deviation at that point
            _, std = self.gpr.predict(state.reshape(1, -1), return_std=True)
            state = np.random.normal(state, std)

            # we skip the sample if it has been selected before;
            # this allows trying different action parameterisations
            if use_sample_buffer and self.sample_selected(state): continue

            updated_state = state_update_fn(state, **kwargs)

            if mode is not None:
                predicate_values = [int(getattr(predicate_library, precondition[0])(**updated_state))
                                    for precondition in self.preconditions[mode]]
            else:
                predicate_values = [int(getattr(predicate_library, precondition[0])(**updated_state))
                                    for precondition in self.preconditions]

            if np.allclose(precondition_values, predicate_values):
                sample_found = True

        if constraints:
            for i, c in enumerate(constraints):
                if c is not None:
                    state[i] = constraints[i]

        if use_sample_buffer:
            self.sample_buffer.append(state)
        return state

    def save(self, model_file_path: str) -> None:
        '''Saves the execution model to the given path (in pickle format).

        Keyword arguments:
        model_file_path: str -- file name for the saved model

        '''
        print('Saving model to {0}'.format(model_file_path))
        with open(model_file_path, 'wb+') as model_file:
            pickle.dump(self, model_file)

    def load(self, model_file_path: str) -> None:
        '''Loads an execution model from the given file (in pickle format).
        Raises an AssertionError if the given file does not exits.

        Keyword arguments:
        model_file_path: str -- file name of the model

        '''
        if not os.path.isfile(model_file_path):
            raise AssertionError('{0} not found'.format(model_file_path))

        print('Loading model from {0}'.format(model_file_path))
        model = None
        with open(model_file_path, 'rb+') as model_file:
            model = pickle.load(model_file)
        self.name = model.name
        self.object_type = model.object_type
        self.preconditions = model.preconditions
        self.gpr = model.gpr

    def sample_selected(self, sample: np.ndarray) -> bool:
        '''Returns True if "sample" exists in the buffer of
        generated samples; returns False otherwise.
        '''
        for s in self.sample_buffer:
            if np.allclose(sample, s):
                return True
        return False

    def diagnose(self, predicate_library: PredicateLibraryBase,
                 state_update_fn: Callable,
                 failure_search_params: FailureSearchParams,
                 mode: str=None,
                 **kwargs):
        '''Looks for diagnosis candidates of a failed action by
        1. reparameterising the action in the vicinity of the parameters that led to a failure and
        2. checking if the reparameterisation leads to a violation of any of the preconditions

        The search is performed by varying the action parameters according to a Gaussian distribution
        (separate distribution for each parameter), such that the standard deviation is increased
        if a diagnosis candidate is not found within a specified number of samples.

        The search procedure may find multiple diagnosis candidates, but only if they violate
        the preconditions within the same search layer.

        Keyword arguments:
        predicate_library: PredicateLibraryBase -- predicate library class used for defining the
                                                   preconditions of the action
        state_update_fn: Callable -- function that predictively applies sampled parameters
                                     so that the preconditions can be verified
        failure_search_params: FailureSearchParams -- parameters used for guiding the process of
                                                      searching for failure diagnoses
        mode: str -- qualitative mode under which the action is executed (default None)

        '''
        assert failure_search_params is not None, 'failure_search_params cannot be None'
        search_params = deepcopy(failure_search_params)

        state = None
        precondition_values = None
        if mode is not None:
            precondition_values = [p[2] for p in self.preconditions[mode]]
        else:
            precondition_values = [p[2] for p in self.preconditions]

        diagnosis_candidates = []
        falsifying_state_updates = {}
        while not diagnosis_candidates:
            sample_count = 0
            for sample_count in range(search_params.max_sample_count):
                state_update = np.zeros(len(search_params.parameter_stds))
                for i, param_std in enumerate(search_params.parameter_stds):
                    state_update[i] = np.random.normal(loc=0., scale=param_std)

                updated_state = state_update_fn(state_update, **kwargs)

                if mode is not None:
                    predicate_values = [int(getattr(predicate_library, precondition[0])(**updated_state))
                                        for precondition in self.preconditions[mode]]
                else:
                    predicate_values = [int(getattr(predicate_library, precondition[0])(**updated_state))
                                        for precondition in self.preconditions]

                if not np.allclose(precondition_values, predicate_values):
                    precondition_values = np.array(precondition_values)
                    predicate_values = np.array(predicate_values)
                    suspect_indices = np.where(np.abs(precondition_values - predicate_values) > 0)[0]

                    for idx in suspect_indices:
                        candidate = (self.preconditions[idx][0],
                                     self.preconditions[idx][1],
                                     predicate_values[idx])
                        if candidate not in diagnosis_candidates:
                            diagnosis_candidates.append(candidate)

                            predicate_name = candidate[0]
                            disjoint_lists = MetaPredicateData.get_disjoint_predicates(predicate_library,
                                                                                       predicate_name)
                            disjoint_diagnoses_count = np.zeros(len(disjoint_lists))
                            for i, disjoint_predicates in enumerate(disjoint_lists):
                                disjoint_diagnoses_count[i] = self.count_predicates_in_candidate_list(disjoint_predicates,
                                                                                                      diagnosis_candidates)

                            # we add the predicate to the list of diagnosis candidates if none of its
                            # disjoint predicates appear in the list already; otherwise, we don't add
                            # the predicate and also remove all disjoint predicates from the list of
                            # current diagnosis candidates;
                            # we additionally update the dictionary of falsitying state updates
                            # with a one-hot vector containing the falsifying update of the parameter
                            # that affects the diagnosis candidate
                            if not np.any(disjoint_diagnoses_count > 1):
                                parameter_cause = MetaPredicateData.get_predicate_parameter_cause(predicate_library,
                                                                                                  candidate[0])
                                parameter_cause_idx = self.parameter_idx_mappings[parameter_cause]
                                falsifying_state_updates[parameter_cause_idx] = state_update[parameter_cause_idx]
                            else:
                                disjoint_predicate_values = {}
                                for k in np.where(disjoint_diagnoses_count>1)[0]:
                                    for p in disjoint_lists[k]:
                                        for c in diagnosis_candidates:
                                            if p == c[0]:
                                                disjoint_predicate_values[p] = c[2]
                                                break

                                contradicting_values = len(disjoint_predicate_values.keys()) != len(set(disjoint_predicate_values.values()))
                                if contradicting_values:
                                    for p in disjoint_predicate_values.keys():
                                        for c in diagnosis_candidates:
                                            if p == c[0]:
                                                diagnosis_candidates.remove(c)

                                                parameter_cause = MetaPredicateData.get_predicate_parameter_cause(predicate_library,
                                                                                                                  candidate[0])
                                                parameter_cause_idx = self.parameter_idx_mappings[parameter_cause]
                                                if parameter_cause_idx in falsifying_state_updates:
                                                    falsifying_state_updates.pop(parameter_cause_idx)
                                                break
                        else:
                            # if the candidate already exists in the list, we update the falsifying update
                            # by taking the one with the smaller values since it provides a more fine-grained falsification
                            parameter_cause = MetaPredicateData.get_predicate_parameter_cause(predicate_library,
                                                                                              candidate[0])
                            parameter_cause_idx = self.parameter_idx_mappings[parameter_cause]
                            if parameter_cause_idx not in falsifying_state_updates:
                                falsifying_state_updates[parameter_cause_idx] = state_update[parameter_cause_idx]
                            elif abs(state_update[parameter_cause_idx]) < abs(falsifying_state_updates[parameter_cause_idx]):
                                falsifying_state_updates[parameter_cause_idx] = state_update[parameter_cause_idx]
            # if the list of candidates is still empty, we update
            # the search radii and repeat the search procedure
            if not diagnosis_candidates:
                for i, param_std in enumerate(search_params.parameter_stds):
                    std_delta = (param_std * search_params.range_increase_percentage) / 100.
                    search_params.parameter_stds[i] = param_std + std_delta

        # the falsifying state update is taken as a sum of the
        # falsifying updates for the individual parameters
        falsifying_state_update = np.zeros(len(search_params.parameter_stds))
        for parameter_idx, update in falsifying_state_updates.items():
            falsifying_state_update[parameter_idx] = update
        return (diagnosis_candidates, falsifying_state_update)

    def count_predicates_in_candidate_list(self, predicate_names: Sequence[str],
                                           diagnosis_candidates: Sequence[Tuple[str, str]]):
        '''Returns the number of times the predicates in "predicate_names"
        appear in "diagnosis_candidate".

        Keyword arguments:
        predicate_names: Sequence[str] -- list of predicate names
        diagnosis_candidates: Sequence[Tuple[str, str]] -- list of (predicate name, predicate parameters)
                                                           tuples containing diagnosis candidates

        '''
        count = 0
        for p in predicate_names:
            for candidate in diagnosis_candidates:
                p_name = candidate
                if type(candidate) is tuple:
                    p_name = candidate[0]

                if p == p_name:
                    count += 1
        return count

    def get_likely_diagnoses(self, predicate_library: PredicateLibraryBase,
                             state_update_fn: Callable,
                             failure_search_params: FailureSearchParams,
                             mode: str=None,
                             diagnosis_repetition_count: int=10,
                             diagnosis_candidate_confidence: float=0.8, **kwargs):
        '''Runs the diagnosis function (self.diagnose) "diagnosis_repetition_count"
        times and only keeps the diagnoses whose appearance ratio is larger than
        "diagnosis_candidate_confidence".

        Returns a tuple (diagnosis_candidates, falsifying_state_update) indicating
        the likely diagnosis candidates and the state update causing the diagnoses.
        The falsifying update is calculated as an average of the updates in which
        the selected diagnosis candidates appear.

        Keyword arguments:
        predicate_library: PredicateLibraryBase -- predicate library class used
                                                   for defining the preconditions
                                                   of the action
        state_update_fn: Callable -- function that predictively applies sampled parameters
                                     so that the preconditions can be verified
        failure_search_params: FailureSearchParams -- parameters used for guiding the
                                                      process of searching for failure diagnoses
        mode: str -- qualitative mode under which the action is executed (default None)
        diagnosis_repetition_count: int -- number of times to generate model violations (default 10)
        diagnosis_candidate_confidence: float -- confidence value for determining that
                                                 model violations are likely failure
                                                 diagnoses (default 0.8)

        '''
        candidate_update_list = []
        diagnosis_candidate_counts = {}
        for _ in range(diagnosis_repetition_count):
            (diagnosis_candidates, falsifying_state_update) = \
                self.diagnose(predicate_library, state_update_fn, failure_search_params, mode, **kwargs)
            candidate_update_list.append((diagnosis_candidates, falsifying_state_update))

            if not diagnosis_candidates:
                continue
            for candidate, _, _ in diagnosis_candidates:
                if not candidate in diagnosis_candidate_counts:
                    diagnosis_candidate_counts[candidate] = 0
                diagnosis_candidate_counts[candidate] += 1

        diagnosis_candidates = []
        falsifying_state_update = np.zeros_like(candidate_update_list[0][1])
        for candidate, count in diagnosis_candidate_counts.items():
            if (count / diagnosis_repetition_count) > diagnosis_candidate_confidence:
                diagnosis_candidates.append(candidate)
                parameter_updates = []
                parameter_cause_idx = -1
                for _, (candidate_list, update) in enumerate(candidate_update_list):
                    for c, _, _ in candidate_list:
                        if candidate == c:
                            parameter_cause = MetaPredicateData.get_predicate_parameter_cause(predicate_library,
                                                                                              candidate)
                            parameter_cause_idx = self.parameter_idx_mappings[parameter_cause]
                            parameter_updates.append(update[parameter_cause_idx])
                            break
                falsifying_state_update[parameter_cause_idx] = np.mean(parameter_updates)

        return (diagnosis_candidates, falsifying_state_update)

    def correct_failed_experience(self, predicate_library: PredicateLibraryBase,
                                  state_update_fn: Callable,
                                  failure_search_params: FailureSearchParams,
                                  mode: str=None,
                                  diagnosis_repetition_count=10,
                                  diagnosis_candidate_confidence=0.8,
                                  correction_sample_count:int =1000, **kwargs):
        '''Suggests a correction of action execution parameters that will make the
        preconditions true under the given qualitative mode. A correction is made by
        1. finding diagnosis candidates of the parameterisation (using self.get_likely_diagnoses) and
        2. suggesting different values for the candidates that are more likely to lead to execution success

        This requires knowledge of which relations are affected by the individual
        action parameters, as parameters that do not affect the diagnosis candidate
        relation should not be changed.

        Keyword arguments:
        predicate_library: PredicateLibraryBase -- predicate library class used
                                                   for defining the preconditions
                                                   of the action
        state_update_fn: Callable -- function that predictively applies sampled parameters
                                     so that the preconditions can be verified
        failure_search_params: FailureSearchParams -- parameters used for guiding the
                                                      process of searching for failure diagnoses
        mode: str -- qualitative mode under which the action is executed (default None)
        diagnosis_repetition_count: int -- number of times to generate model violations (default 10)
        diagnosis_candidate_confidence: float -- confidence value for determining that
                                                 model violations are likely failure
                                                 diagnoses (default 0.8)
        correction_sample_count: int (default 1000)

        '''
        (diagnosis_candidates, falsifying_state_update) = self.get_likely_diagnoses(predicate_library,
                                                                                    state_update_fn,
                                                                                    failure_search_params,
                                                                                    mode,
                                                                                    diagnosis_repetition_count,
                                                                                    diagnosis_candidate_confidence,
                                                                                    **kwargs)

        precondition_values = None
        if mode is not None:
            precondition_values = [p[2] for p in self.preconditions[mode]]
        else:
            precondition_values = [p[2] for p in self.preconditions]

        # we generate state update candidates by moving in the parameter direction
        # opposite the one of the falsification; the samples are generated from a
        # Gamma distribution scaled by the parameter update that falsifies
        # a predicate and shaped by an arbitrary number
        correction_samples = np.zeros((correction_sample_count, len(falsifying_state_update)))
        for candidate in diagnosis_candidates:
            parameter_cause = MetaPredicateData.get_predicate_parameter_cause(predicate_library,
                                                                              candidate)
            parameter_cause_idx = self.parameter_idx_mappings[parameter_cause]

            gamma_samples = np.random.gamma(shape=2.,
                                            scale=abs(falsifying_state_update[parameter_cause_idx]),
                                            size=correction_sample_count)
            falsifying_sign = np.sign(falsifying_state_update[parameter_cause_idx])
            correction_sign = falsifying_sign * -1.
            correction_samples[:, parameter_cause_idx] = gamma_samples * correction_sign

        # we filter out the samples that do not satisfy the preconditions;
        # we use the GP to calculate the success probabilities (and the
        # uncertainties) of the samples that do
        filtered_samples = []
        sample_success_likelihoods = []
        sample_success_uncertainties = []
        for sample in correction_samples:
            updated_state = state_update_fn(sample, **kwargs)

            if mode is not None:
                predicate_values = [int(getattr(predicate_library, precondition[0])(**updated_state))
                                    for precondition in self.preconditions[mode]]
            else:
                predicate_values = [int(getattr(predicate_library, precondition[0])(**updated_state))
                                    for precondition in self.preconditions]

            if np.allclose(precondition_values, predicate_values):
                success_p, std = self.gpr.predict(sample.reshape(1,-1), return_std=True)
                filtered_samples.append(sample)
                sample_success_likelihoods.append(success_p)
                sample_success_uncertainties.append(std)

        # we return the sample with the highest predicted success
        # as a corrected execution sample, or a zero array
        # of the samples survived the filtering stage
        if not filtered_samples: return (diagnosis_candidates, np.zeros(len(falsifying_state_update)))

        max_weight_idx = np.argmax(sample_success_likelihoods)
        return (diagnosis_candidates, filtered_samples[max_weight_idx])
