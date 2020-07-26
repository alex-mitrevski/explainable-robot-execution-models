import os

import pickle
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import NearestNeighbors

class ActionExecutionModel(object):
    # action name
    name = None

    # action preconditions
    preconditions = None

    # Gaussian process regressor
    gpr = None

    def __init__(self, name, model_file_path=None):
        self.name = name
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

    def sample_state(self, value, predicate_library, state_update_fn,
                     mode=None, **kwargs):
        '''Returns a state for which the underlying GP equals "value".

        Keyword arguments:
        value: int -- function value for which an state should be sampled (default 1)

        '''
        state = None
        precondition_values = None
        if mode is not None:
            precondition_values = [p[2] for p in self.preconditions[mode]]
        else:
            precondition_values = [p[2] for p in self.preconditions]

        sample_found = False
        value_indices = np.where(self.gpr.y_train_==value)[0]
        constraints = kwargs.get('constraints', None)

        if constraints:
            non_constrained_param_idx = [i for i, cx in enumerate(constraints)
                                         if cx is None]
            constrained_training_data = np.array(self.gpr.X_train_[value_indices])
            constrained_training_data[:,non_constrained_param_idx] = 0.

            n_neighbours = int(self.gpr.X_train_.shape[0] * 0.1)
            n_neighbours = n_neighbours if n_neighbours < constrained_training_data.shape[0] \
                                        else constrained_training_data.shape[0]
            neighbour_tree = NearestNeighbors(n_neighbors=n_neighbours,
                                              algorithm='ball_tree').fit(constrained_training_data)

            constrained_instance = np.array([c if c is not None else 0.
                                             for c in constraints])[np.newaxis]
            distances, indices = neighbour_tree.kneighbors(constrained_instance)

        while not sample_found:
            # we take a random input from the training set
            # where the function is equal to "value"
            random_state_idx = np.random.choice(value_indices)
            state = np.array(self.gpr.X_train_[random_state_idx])

            # the returned state is sampled from a normal
            # distribution centered at the selected input
            # and with a standard deviation equal to
            # the GP standard deviation at that point
            _, std = self.gpr.predict(state.reshape(1,-1), return_std=True)
            state = np.random.normal(state, std)
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
        return state

    def save(self, model_file_path):
        '''Saves the execution model to the given path (in pickle format).

        Keyword arguments:
        model_file_path: str -- file name for the saved model

        '''
        print('Saving model to {0}'.format(model_file_path))
        with open(model_file_path, 'wb+') as model_file:
            pickle.dump(self, model_file)

    def load(self, model_file_path):
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
        self.preconditions = model.preconditions
        self.gpr = model.gpr
