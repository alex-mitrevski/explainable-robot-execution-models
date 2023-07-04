import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from action_execution.geometry.vector import Vector3
from action_execution.geometry.pose import Pose3
from action_execution.geometry.bbox import BBox3

from robot_action_execution.action_execution_model import ActionExecutionModel
from robot_action_execution.predicate_learner import RuleLearner
from robot_action_execution.predicates.predicate_utils import MetaPredicateData
from robot_action_execution.predicates.object_grasping import ObjectGraspingPredicateLibrary

class GraspObjectExecutionModel(ActionExecutionModel):
    def learn_preconditions(self, data_point_count: int,
                            goal_positions: np.ndarray,
                            goal_orientations: np.ndarray,
                            object_positions: np.ndarray,
                            object_orientations: np.ndarray,
                            object_bb_mins: np.ndarray,
                            object_bb_maxs: np.ndarray,
                            labels: np.ndarray) -> None:
        success_indices = np.where(labels==1)[0]
        predicate_values = self.extract_predicate_values(data_point_count,
                                                         goal_positions, goal_orientations,
                                                         object_positions, object_orientations,
                                                         object_bb_mins, object_bb_maxs)
        self.preconditions = self.get_preconditions(predicate_values, success_indices)

    def learn_success_model(self, action_parameters: np.ndarray,
                            labels: np.ndarray) -> None:
        kernel = C(1., (0.1, 100)) * RBF(1., (1e-2, 1e2))
        gpr = GaussianProcessRegressor(kernel=kernel).fit(action_parameters, labels)
        self.gpr = gpr

    def extract_predicate_values(self, data_point_count: int,
                                 goal_positions: np.ndarray,
                                 goal_orientations: np.ndarray,
                                 object_positions: np.ndarray,
                                 object_orientations: np.ndarray,
                                 object_bb_mins: np.ndarray,
                                 object_bb_maxs: np.ndarray) -> np.ndarray:
        predicate_values = np.zeros((data_point_count,
                                     MetaPredicateData.get_predicate_count(ObjectGraspingPredicateLibrary)),
                                    dtype=int)
        predicates = MetaPredicateData.get_predicates(ObjectGraspingPredicateLibrary)
        predicate_names = MetaPredicateData.get_predicate_names(ObjectGraspingPredicateLibrary)
        for i in range(data_point_count):
            gripper_position = Vector3(x=goal_positions[i,0],
                                       y=goal_positions[i,1],
                                       z=goal_positions[i,2])
            gripper_orientation = Vector3(x=goal_orientations[i,0],
                                          y=goal_orientations[i,1],
                                          z=goal_orientations[i,2])
            gripper_pose = Pose3(position=gripper_position,
                                 orientation=gripper_orientation)

            object_position = Vector3(x=object_positions[i,0],
                                      y=object_positions[i,1],
                                      z=object_positions[i,2])
            object_orientation = Vector3(x=object_orientations[i,0],
                                         y=object_orientations[i,1],
                                         z=object_orientations[i,2])
            object_pose = Pose3(position=object_position,
                                orientation=object_orientation)

            bb_min = Vector3(x=object_bb_mins[i,0],
                             y=object_bb_mins[i,1],
                             z=object_bb_mins[i,2])
            bb_max = Vector3(x=object_bb_maxs[i,0],
                             y=object_bb_maxs[i,1],
                             z=object_bb_maxs[i,2])
            object_bbox = BBox3(min_values=bb_min, max_values=bb_max)

            for j, p in enumerate(predicates):
                if predicate_names[j].find('parallel') != -1 or predicate_names[j].find('perpendicular') != -1:
                    predicate_values[i,j] = p(gripper_pose, object_pose)
                else:
                    predicate_values[i,j] = p(gripper_pose, object_bbox)

        return predicate_values

    def get_preconditions(self, predicate_values, success_idx):
        predicate_strings = [('{0}'.format(p), ['pose', 'b_box'])
                            if p.find('parallel') == -1 and p.find('perpendicular') == -1
                            else ('{0}'.format(p), ['pose', 'object_pose'])
                            for p in MetaPredicateData.get_predicate_names(ObjectGraspingPredicateLibrary)]
        precondition_vector, precondition_values = RuleLearner.learn_rules(predicate_values[success_idx],
                                                                           predicate_acceptance_threshold=0.1)
        preconditions = RuleLearner.extract_preconditions(precondition_vector,
                                                          precondition_values,
                                                          predicate_strings)
        return preconditions
