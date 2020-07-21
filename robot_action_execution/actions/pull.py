import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from action_execution.geometry.vector import Vector3
from action_execution.geometry.pose import Pose3
from action_execution.geometry.bbox import BBox3
from action_execution.extern import transformations as tf

from robot_action_execution.action_execution_model import ActionExecutionModel
from robot_action_execution.predicate_learner import RuleLearner
from robot_action_execution.predicates.predicate_utils import MetaPredicateData
from robot_action_execution.predicates.pulling import PullingPredicateLibrary

class PullExecutionModel(ActionExecutionModel):
    def learn_preconditions(self, data_point_count: int,
                            handle_positions: np.ndarray,
                            goal_positions: np.ndarray,
                            motion_durations: np.ndarray,
                            labels: np.ndarray) -> None:
        success_indices = np.where(labels==1)[0]
        predicate_values = self.extract_predicate_values(data_point_count, handle_positions,
                                                         goal_positions, motion_durations)

        p_short = self.get_preconditions_for_qualitative_mode(predicate_values,
                                                              handle_positions,
                                                              goal_positions,
                                                              success_indices,
                                                              PullingPredicateLibrary.SHORT_DISTANCE)
        p_long = self.get_preconditions_for_qualitative_mode(predicate_values,
                                                             handle_positions,
                                                             goal_positions,
                                                             success_indices,
                                                             PullingPredicateLibrary.LONG_DISTANCE)

        self.preconditions[PullingPredicateLibrary.SHORT_DISTANCE] = p_short
        self.preconditions[PullingPredicateLibrary.LONG_DISTANCE] = p_long

    def learn_success_model(self, action_parameters: np.ndarray,
                            labels: np.ndarray) -> None:
        kernel = C(1., (0.1, 100)) * RBF(1., (1e-2, 1e2))
        gpr = GaussianProcessRegressor(kernel=kernel).fit(action_parameters, labels)
        self.gpr = gpr

    def extract_predicate_values(self, data_point_count: int,
                                 handle_positions: np.ndarray,
                                 goal_positions: np.ndarray,
                                 motion_durations: np.ndarray) -> np.ndarray:
        predicate_values = np.zeros((data_point_count,
                                     MetaPredicateData.get_predicate_count(PullingPredicateLibrary) + 1),
                                    dtype=int)
        predicates = MetaPredicateData.get_predicates(PullingPredicateLibrary)
        for i in range(data_point_count):
            handle_position = Vector3(x=handle_positions[i, 0],
                                      y=handle_positions[i, 1],
                                      z=handle_positions[i, 2])
            handle_orientation = Vector3(x=0., y=0., z=0.)
            handle_pose = Pose3(position=handle_position,
                                orientation=handle_orientation)

            goal_position = Vector3(x=goal_positions[i, 0],
                                    y=goal_positions[i, 1],
                                    z=goal_positions[i, 2])
            goal_orientation = Vector3(x=0., y=0., z=0.)
            goal_pose = Pose3(position=goal_position,
                              orientation=goal_orientation)

            for j, p in enumerate(predicates):
                predicate_values[i,j] = p(handle_pose, goal_pose)

            predicate_values[i,-1] = PullingPredicateLibrary.qualitative_time(motion_durations[i])
        return predicate_values

    def get_preconditions_for_qualitative_mode(self, predicate_values,
                                               handle_positions,
                                               goal_positions,
                                               success_idx,
                                               desired_qualitative_distance):
        qualitative_distances = np.array([PullingPredicateLibrary.qualitative_distance(d) for d
                                          in np.linalg.norm(handle_positions - goal_positions, axis=1)], dtype=int)

        distance_idx = np.where(qualitative_distances == desired_qualitative_distance)[0]
        distance_success_idx = list(set(distance_idx).intersection(success_idx))

        precondition_vector, precondition_values = RuleLearner.learn_rules(predicate_values[distance_success_idx],
                                                                           predicate_acceptance_threshold=0.1)

        predicate_strings = [('{0}'.format(p), ['object', 'goal'])
                             for p in MetaPredicateData.get_predicate_names(PullingPredicateLibrary)]
        predicate_strings.append((('qualitative_time', ['t'])))
        preconditions = RuleLearner.extract_preconditions(precondition_vector,
                                                          precondition_values,
                                                          predicate_strings)

        return preconditions
