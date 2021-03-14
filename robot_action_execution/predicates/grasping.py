from action_execution.geometry.pose import Pose3
from action_execution.geometry.bbox import BBox3

from robot_action_execution.predicates.predicate_utils import PredicateLibraryBase

class GraspingPredicateLibrary(PredicateLibraryBase):
    relation_names = ['in_front_of_x', 'behind_x',
                      'in_front_of_y', 'behind_y',
                      'above', 'below', 'centered_along_x',
                      'centered_along_y', 'centered_along_z']

    @staticmethod
    def in_front_of_x(pose: Pose3, b_box: BBox3) -> bool:
        return pose.position.x < b_box.min.x

    @staticmethod
    def behind_x(pose: Pose3, b_box: BBox3) -> bool:
        return pose.position.x > b_box.max.x

    @staticmethod
    def in_front_of_y(pose: Pose3, b_box: BBox3) -> bool:
        return pose.position.y < b_box.min.y

    @staticmethod
    def behind_y(pose: Pose3, b_box: BBox3) -> bool:
        return pose.position.y > b_box.max.y

    @staticmethod
    def above(pose: Pose3, b_box: BBox3) -> bool:
        return pose.position.z > b_box.max.z

    @staticmethod
    def below(pose: Pose3, b_box: BBox3) -> bool:
        return pose.position.z < b_box.min.z

    @staticmethod
    def centered_along_x(pose: Pose3, b_box: BBox3) -> bool:
        bbox_center = (b_box.min.x + b_box.max.x) / 2.
        return abs(pose.position.x - bbox_center) < 0.05

    @staticmethod
    def centered_along_y(pose: Pose3, b_box: BBox3) -> bool:
        bbox_center = (b_box.min.y + b_box.max.y) / 2.
        return abs(pose.position.y - bbox_center) < 0.05

    @staticmethod
    def centered_along_z(pose: Pose3, b_box: BBox3) -> bool:
        bbox_center = (b_box.min.z + b_box.max.z) / 2.
        return abs(pose.position.z - bbox_center) < 0.05
