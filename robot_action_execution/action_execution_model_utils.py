from typing import Sequence, List, Tuple
import numpy as np

from black_box_tools.db_utils import DBUtils
from mas_knowledge_utils.ontology_query_interface import OntologyQueryInterface

from robot_action_execution.action_execution_model import ActionExecutionModel

class FailureSearchParams(object):
    '''Parameters used to define a search for failure diagnoses. For more details
    about what this means, please see

    A. Mitrevski, P. G. Plöger, and G. Lakemeyer, "Robot Action Diagnosis and Experience Correction by Falsifying Parameterised Execution Models,"
    in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2021.
    '''
    # parameter standard deviations defining a failure search region
    # (each parameter has its own standard deviation in order to
    # take into account different parameter ranges)
    parameter_stds = None

    # maximum number of samples to be generated within one search region
    max_sample_count = 0

    # percentage by which to increase the search region for failure diagnoses
    # if no diagnoses are found within the current region
    range_increase_percentage = 0.

    def __init__(self, parameter_stds: Sequence[float]=None,
                 max_sample_count: int=0,
                 range_increase_percentage: float=0.):
        self.parameter_stds = list(parameter_stds) if parameter_stds is not None else None
        self.max_sample_count = max_sample_count
        self.range_increase_percentage = range_increase_percentage

    def __deepcopy__(self, memo):
        return FailureSearchParams(self.parameter_stds,
                                   self.max_sample_count,
                                   self.range_increase_percentage)

    def __repr__(self):
        return 'parameter_stds: {0}\nmax_sample_count: {1}\nrange_increase_percentage: {2}'.format(self.parameter_stds,
                                                                                                   self.max_sample_count,
                                                                                                   self.range_increase_percentage)


class GeneralisationData(object):
    '''Parameters used for execution model generalisation between objects.

    For more details about what this generalisation means, please take a look at the following paper:

    A. Mitrevski, P. G. Plöger, and G. Lakemeyer, "Ontology-Assisted Generalisation of Robot Action Execution Knowledge,"
    Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021.

    Generalisation data are stored in a dedicated collection in a MongoDB database; this
    component thus writes generalisation data updates to the database.

    Generalisation data are stored in the collection as action-specific documents.
    Each of these action-specific documents includes:
    * the name of the action
    * the prior values of the Beta distribution used for estimating
      the success probability of a given generalisation
    * objects for which:
        - either an execution model is known (a pkl file can be loaded for these) or
        - a model is not known and for which generalisation data is stored, consisting of
          the objects whose models have been attempted, the number of total generalisation
          attempts and number of successes, as well as the calculated suitability
    The general format of the action-specific document is shown below:

    {
        "action_name" : str,
        "success_probability_parameters" : {
            "alpha_prior" : float,
            "beta_prior" : float
        },
        "objects" : {
            "Obj_1" : {
                "load" : "model-to-load.pkl"
            },
            ...
            "Obj_k" : {
                "generalisations" : {
                    "Obj1" : {
                        "total_attempts" : int,
                        "successful_attempts" : int,
                        "suitability" : float
                    },
                    ...
                }
            },
            ...
        }
    }

    '''
    # type of object whose model is being generalised
    generalising_object = None

    # type of object that a model is being generalised to
    obj_generalised_to = None

    # name of the executing action
    action_name = None

    # total number of attempted generalisations for executing action "action_name"
    # with the object "obj_generalised_to" using the model of "generalising_object"
    total_attempts = 0

    # number of successes when executing action "action_name" with the object
    # "obj_generalised_to" using the model of "generalising_object"
    successful_attempts = 0

    # suitability of the model of "generalising_object" for executing action "action_name" with
    # the object "obj_generalised_to" (please see the above paper for a definition of suitability)
    suitability = -1.

    # prior alpha value of the Beta distribution used for estimating the success probability
    alpha_prior = 0

    # prior beta value of the Beta distribution used for estimating the success probability
    beta_prior = 0

    # number of samples to use in a Beta distribution mean estimator
    SAMPLES_FOR_SUCCESS_ESTIMATION = 100

    def __init__(self, generalising_object: str, obj_generalised_to: str, generalisation_data_dict):
        self.generalising_object = generalising_object
        self.obj_generalised_to = obj_generalised_to
        self.action_name = generalisation_data_dict['action_name']
        self.alpha_prior = generalisation_data_dict['success_probability_parameters']['alpha_prior']
        self.beta_prior = generalisation_data_dict['success_probability_parameters']['beta_prior']

        if obj_generalised_to not in generalisation_data_dict['objects']:
            return

        for obj, generalisation in generalisation_data_dict['objects'][obj_generalised_to]['generalisations'].items():
            if obj == generalising_object:
                self.total_attempts = generalisation['total_attempts']
                self.successful_attempts = generalisation['successful_attempts']
                self.suitability = generalisation['suitability']
                break

    def get_success_probability(self) -> float:
        '''Returns the estimated success probability of using the model of self.generalising_object
        for executing action self.action_name with the object self.obj_generalised_to.
        '''
        alpha, beta = self.get_beta_params()
        beta_samples = np.random.beta(alpha, beta, GeneralisationData.SAMPLES_FOR_SUCCESS_ESTIMATION)
        success_probability = np.mean(beta_samples)
        return success_probability

    def get_beta_params(self) -> Tuple[float, float]:
        '''Returns posterior Beta distribution parameters (alpha and beta)
        given the number of generalisation successes and failures.
        '''
        successes = self.successful_attempts
        failures = self.total_attempts - self.successful_attempts
        alpha = self.alpha_prior + successes - 1
        beta = self.beta_prior + failures - 1
        return alpha, beta

    def save_updated_suitability(self, suitability: float,
                                 log_db_name: str='action_execution',
                                 generalisation_collection_name: str='suitability_graph') -> None:
        '''Stores the given suitability value in the given MondoDB collection.

        The function takes care of inserting new entries if:
        * there is no prior generalisation data for self.obj_generalised_to
        * there are no prior generaliation attempts of using the model of self.generalising_object
        for executing action self.action_name with the object self.obj_generalised_to

        Keyword arguments:
        suitability: float -- posterior suitability
        log_db_name: str   -- name of a MongoDB database where the updated suitabilities
                              should be stored (default "action_execution")
        generalisation_collection_name: str -- name of a MongoDB collection in which
                                               generalisation data are stored
                                               (default "suitability_graph")

        '''
        self.suitability = suitability

        mongo_client = DBUtils.get_db_client()
        db = mongo_client[log_db_name]
        collection = db[generalisation_collection_name]
        generalisation_data_dict = collection.find_one({'action_name': self.action_name})

        if self.obj_generalised_to not in generalisation_data_dict['objects']:
            generalisation_data_dict['objects'][self.obj_generalised_to] = {'generalisations': {}}
            generalisations_dict = {}
            generalisations_dict[self.generalising_object] = {'total_attempts': 0,
                                                              'successful_attempts': 0,
                                                              'suitability': suitability}
            generalisation_data_dict['objects'][self.obj_generalised_to]['generalisations'] = generalisations_dict
        else:
            if self.generalising_object not in generalisation_data_dict['objects'][self.obj_generalised_to]['generalisations']:
                generalisations_dict = {'total_attempts': 0,
                                        'successful_attempts': 0,
                                        'suitability': suitability}
                generalisation_data_dict['objects'][self.obj_generalised_to]['generalisations'][self.generalising_object] = generalisations_dict
            else:
                for generalising_obj in generalisation_data_dict['objects'][self.obj_generalised_to]['generalisations']:
                    if generalising_obj == self.generalising_object:
                        generalisation_data_dict['objects'][self.obj_generalised_to]['generalisations'][self.generalising_object]['suitability'] = suitability
                        break

        collection.replace_one({'action_name': self.action_name}, generalisation_data_dict)

    def save_updated_generalisation_data(self, execution_success: bool,
                                         log_db_name: str='action_execution',
                                         generalisation_collection_name: str='suitability_graph') -> None:
        '''Updates the number of total generalisation attempts and the number of
        generalision successes of using the model of self.generalising_object
        for executing action self.action_name with the object self.obj_generalised_to.

        Keyword arguments:
        execution_success: bool -- whether the execution in the last generalisation
                                   attempt was successful
        log_db_name: str   -- name of a MongoDB database where the updated data
                              should be stored (default "action_execution")
        generalisation_collection_name: str -- name of a MongoDB collection in which
                                               generalisation data are stored
                                               (default "suitability_graph")

        '''
        self.total_attempts += 1
        if execution_success:
            self.successful_attempts += 1

        mongo_client = DBUtils.get_db_client()
        db = mongo_client[log_db_name]
        collection = db[generalisation_collection_name]
        generalisation_data_dict = collection.find_one({'action_name': self.action_name})

        for generalising_obj in generalisation_data_dict['objects'][self.obj_generalised_to]['generalisations']:
            if generalising_obj == self.generalising_object:
                generalisation_data_dict['objects'][self.obj_generalised_to]['generalisations'][self.generalising_object]['total_attempts'] += 1
                if execution_success:
                    generalisation_data_dict['objects'][self.obj_generalised_to]['generalisations'][self.generalising_object]['successful_attempts'] += 1
                break

        collection.replace_one({'action_name': self.action_name}, generalisation_data_dict)

    def __str__(self):
        obj_str = ''
        obj_str += 'generalising_object: {0}\n'.format(self.generalising_object)
        obj_str += 'obj_generalised_to: {0}\n'.format(self.obj_generalised_to)
        obj_str += 'total_attempts: {0}\n'.format(self.total_attempts)
        obj_str += 'successful_attempts: {0}\n'.format(self.successful_attempts)
        obj_str += 'suitability: {0}\n'.format(self.suitability)
        obj_str += 'success_probability_parameters:\n'
        obj_str += '  alpha_prior: {0}\n'.format(self.alpha_prior)
        obj_str += '  beta_prior: {0}\n'.format(self.beta_prior)
        return obj_str

    def __repr__(self):
        return str(self)


class ActionExecutionModelUtils(object):
    '''Execution model utilities dealing with model generalisation between objects.
    For more details about what this generalisation means, please take a look at the following paper:

    A. Mitrevski, P. G. Plöger, and G. Lakemeyer, "Ontology-Assisted Generalisation of Robot Action Execution Knowledge,"
    Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021.

    '''
    @staticmethod
    def get_related_objects(obj_class: str,
                            ontology_interface: OntologyQueryInterface) -> List[str]:
        '''Returns a list of classes that form an object cluster in the ontology.

        Keyword arguments:
        obj_class: str -- name of an object class in the ontology
        ontology_interface: mas_knowledge_utils.ontology_query_interface.OntologyQueryInterface
                          -- instance of an ontology query interface
        '''
        ancestor_hierarchy = ontology_interface.get_ancestor_hierarchy(obj_class)
        ancestors = []
        for level in ancestor_hierarchy:
            ancestors.extend(level)

        parents = ontology_interface.get_parent_classes_of(obj_class, only_parents=True)
        siblings = sum([ontology_interface.get_subclasses_of(parent, only_children=True)
                        for parent in parents], [])
        siblings.remove(obj_class)
        children = ontology_interface.get_subclasses_of(obj_class, only_children=True)

        related_objects = ancestors + siblings + children
        return related_objects

    @staticmethod
    def get_execution_model(action_name: str, obj_type: str,
                            ontology_interface: OntologyQueryInterface,
                            action_mode: str=None,
                            verbose: bool=False) -> ActionExecutionModel:
        '''Returns an execution model instance for executing the given action
        with an object of the given type.

        Keyword argument:
        action_name: str -- name of an action being executed
        obj_type: str -- type of the object for which the action is being executed
        ontology_interface: mas_knowledge_utils.ontology_query_interface.OntologyQueryInterface
                          -- instance of an ontology query interface
        action_mode: str -- action mode under which the action is executed -
                            currently unused in the function (default None)
        verbose: bool -- whether to print some debugging messages (default False)

        '''
        # if a model for the given object class is already known, we simply return that one
        obj_model = ActionExecutionModelUtils.get_model(action_name, obj_type)
        if obj_model is not None:
            return obj_model

        # if a model for the given object class is not known, we try to find
        # an appropriate existing model to generalise
        related_objects = ActionExecutionModelUtils.get_related_objects(obj_type, ontology_interface)
        object_cluster = []
        for obj in related_objects:
            model = ActionExecutionModelUtils.get_model(action_name, obj)
            if model is not None:
                object_cluster.append((obj, model))

        if verbose:
            print('Object cluster: {0}'.format([obj for obj, _ in object_cluster]))

        posteriors = np.zeros(len(object_cluster))
        normaliser = 0.
        generalisation_data_list = []
        for i, (related_obj, _) in enumerate(object_cluster):
            object_similarity = ontology_interface.class_similarity(obj_type, related_obj)
            generalisation_data = ActionExecutionModelUtils.get_model_generalisation_data(action_name,
                                                                                          obj_type,
                                                                                          related_obj)
            success_probability = generalisation_data.get_success_probability()

            prior_suitability = generalisation_data.suitability
            if prior_suitability < 0.:
                prior_suitability = 1. / len(object_cluster)

            posteriors[i] = object_similarity * success_probability * prior_suitability
            normaliser += posteriors[i]
            generalisation_data_list.append(generalisation_data)

        posteriors /= normaliser
        max_posterior_idx = np.argmax(posteriors)
        _, max_posterior_model = object_cluster[max_posterior_idx]

        for i, generalisation_data in enumerate(generalisation_data_list):
            generalisation_data.save_updated_suitability(posteriors[i])
            if verbose:
                print('{0}: {1}'.format(generalisation_data.generalising_object, posteriors[i]))
        if verbose:
            print()
        return max_posterior_model

    @staticmethod
    def get_model(action_name: str, obj_type: str,
                  log_db_name: str='action_execution') -> ActionExecutionModel:
        '''Returns an execution model if one exists for the given action
        and object class; returns None otherwise.

        Keyword arguments:
        action_name: str -- name of an action being executed
        obj_type: str -- type of the object for which the action is being executed
        log_db_name: str   -- name of a MongoDB database where model generalisation
                              data are stored (default "action_execution")

        '''
        action_doc = ActionExecutionModelUtils.get_action_generalisation_doc(action_name, log_db_name)
        if obj_type in action_doc['objects'] and 'load' in action_doc['objects'][obj_type]:
            model_path = action_doc['objects'][obj_type]['load']
            return ActionExecutionModel(action_name, obj_type, model_path)
        return None

    @staticmethod
    def get_model_generalisation_data(action_name: str,
                                      obj_generalised_to: str,
                                      generalising_obj: str,
                                      log_db_name: str='action_execution') -> GeneralisationData:
        '''Returns a generalisation data instance which contains data about
        generalisation attempts for executing action_name for objects of type
        obj_generalised_to with a model that is known to perform well for
        objects of type generalising_obj.

        Keyword arguments:
        action_name: str -- name of an action being executed
        obj_generalised_to: str -- type of object that a model is being generalised to
        generalising_obj: str -- type of object whose model is being generalised
        log_db_name: str   -- name of a MongoDB database where model generalisation
                              data are stored (default "action_execution")

        '''
        action_doc = ActionExecutionModelUtils.get_action_generalisation_doc(action_name, log_db_name)
        generalisation_data = GeneralisationData(generalising_object=generalising_obj,
                                                 obj_generalised_to=obj_generalised_to,
                                                 generalisation_data_dict=action_doc)
        return generalisation_data

    @staticmethod
    def update_model_generalisation_attempts(action_name: str,
                                             obj_generalised_to: str,
                                             generalising_obj: str,
                                             execution_success: bool,
                                             log_db_name: str='action_execution') -> None:
        '''Updates the generalisation data of using the model of generalising_obj
        for executing action action_name with the object obj_generalised_to.

        Keyword arguments:
        action_name: str -- name of an action being executed
        obj_generalised_to: str -- type of object that a model is being generalised to
        generalising_obj: str -- type of object whose model is being generalised
        execution_success: bool -- whether the execution in the last generalisation
                                   attempt was successful
        log_db_name: str   -- name of a MongoDB database where model generalisation
                              data are stored (default "action_execution")

        '''
        generalisation_data = ActionExecutionModelUtils.get_model_generalisation_data(action_name,
                                                                                      obj_generalised_to,
                                                                                      generalising_obj,
                                                                                      log_db_name)
        generalisation_data.save_updated_generalisation_data(execution_success)

    @staticmethod
    def get_action_generalisation_doc(action_name: str,
                                      log_db_name: str='action_execution',
                                      generalisation_collection_name: str='suitability_graph'):
        '''Returns a dictionary representing a MongoDB document with
        generalisation data about the given action.

        Keyword arguments:
        action_name: str -- name of an action being executed
        log_db_name: str   -- name of a MongoDB database where model generalisation
                              data are stored (default "action_execution")
        generalisation_collection_name: str -- name of a MongoDB collection in which
                                               generalisation data are stored
                                               (default "suitability_graph")

        '''
        generalisation_docs = DBUtils.get_all_docs(log_db_name, generalisation_collection_name)
        action_doc = None
        for doc in generalisation_docs:
            if doc['action_name'] == action_name:
                action_doc = doc
                break
        return action_doc
