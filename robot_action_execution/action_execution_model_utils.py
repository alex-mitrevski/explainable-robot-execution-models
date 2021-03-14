from typing import Sequence

class FailureSearchParams(object):
    '''Parameters used to define a search for failure diagnoses. For more details
    about what this means, please see

    A. Mitrevski, P. G. Plöger, and G. Lakemeyer, "Robot Action Diagnosis and Experience Correction by Falsifying Parameterised Execution Models," in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2021.
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
