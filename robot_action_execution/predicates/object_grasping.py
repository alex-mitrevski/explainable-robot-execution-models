from action_execution.geometry.pose import Pose3
from action_execution.geometry.bbox import BBox3

from robot_action_execution.predicates.predicate_utils import PredicateLibraryBase

class ObjectGraspingPredicateLibrary(PredicateLibraryBase):
    relation_names = ['in_front_of_x', 'far_in_front_of_x', 'behind_x',
                      'in_front_of_y', 'behind_y',
                      'above', 'below', 'centered_along_x',
                      'centered_along_y', 'centered_along_z',
                      'parallel_along_z', 'perpendicular_along_z']

    relation_parameter_causes = {'pose.position.x': ['in_front_of_x', 'far_in_front_of_x', 'behind_x', 'centered_along_x'],
                                 'pose.position.y': ['in_front_of_y', 'behind_y', 'centered_along_y'],
                                 'pose.position.z': ['above', 'below', 'centered_along_z']}

    disjoint_predicates = [('in_front_of_x', 'behind_x', 'centered_along_x'),
                           ('far_in_front_of_x', 'behind_x', 'centered_along_x'),
                           ('in_front_of_y', 'behind_y', 'centered_along_y'),
                           ('above', 'below', 'centered_along_z'),
                           ('parallel_along_z', 'perpendicular_along_z')]

    @staticmethod
    def in_front_of_x(pose: Pose3, b_box: BBox3) -> bool:
        return pose.position.x < b_box.min.x

    @staticmethod
    def far_in_front_of_x(pose: Pose3, b_box: BBox3, eps: float=0.05) -> bool:
        return (pose.position.x < b_box.min.x) and (b_box.min.x - pose.position.x) > eps

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
    def centered_along_x(pose: Pose3, b_box: BBox3, eps: float=0.05) -> bool:
        bbox_center = (b_box.min.x + b_box.max.x) / 2.
        return abs(pose.position.x - bbox_center) < eps and \
               not ObjectGraspingPredicateLibrary.behind_x(pose, b_box) and \
               not ObjectGraspingPredicateLibrary.in_front_of_x(pose, b_box)

    @staticmethod
    def centered_along_y(pose: Pose3, b_box: BBox3, eps: float=0.05) -> bool:
        bbox_center = (b_box.min.y + b_box.max.y) / 2.
        return abs(pose.position.y - bbox_center) < eps and \
               not ObjectGraspingPredicateLibrary.behind_y(pose, b_box) and \
               not ObjectGraspingPredicateLibrary.in_front_of_y(pose, b_box)

    @staticmethod
    def centered_along_z(pose: Pose3, b_box: BBox3, eps: float=0.05) -> bool:
        bbox_center = (b_box.min.z + b_box.max.z) / 2.
        return abs(pose.position.z - bbox_center) < eps and \
               not ObjectGraspingPredicateLibrary.above(pose, b_box) and \
               not ObjectGraspingPredicateLibrary.below(pose, b_box)

    @staticmethod
    def parallel_along_z(pose: Pose3,
                         object_pose: Pose3,
                         eps: float=0.435 # ~25 degrees
                         ) -> bool:
        return abs(pose.orientation.z - object_pose.orientation.z) < eps

    @staticmethod
    def perpendicular_along_z(pose: Pose3,
                              object_pose: Pose3,
                              eps: float=0.435 # ~25 degrees
                              ) -> bool:
        return abs(abs(pose.orientation.z - object_pose.orientation.z) - (np.pi / 2)) < eps