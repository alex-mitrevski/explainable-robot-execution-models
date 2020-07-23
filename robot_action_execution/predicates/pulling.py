from action_execution.geometry.pose import Pose3

class PullingPredicateLibrary(object):
    relation_names = ['in_front_of_x', 'behind_x',
                      'in_front_of_y', 'behind_y',
                      'above', 'below', 'centered_along_x',
                      'centered_along_y', 'centered_along_z']

    SHORT_DISTANCE = 0
    LONG_DISTANCE = 1

    SHORT_DURATION = 0
    LONG_DURATION = 1

    @staticmethod
    def in_front_of_x(pose1: Pose3, pose2: Pose3, **kwargs) -> bool:
        return pose1.position.x < pose2.position.x

    @staticmethod
    def behind_x(pose1: Pose3, pose2: Pose3, **kwargs) -> bool:
        return pose1.position.x > pose2.position.x

    @staticmethod
    def in_front_of_y(pose1: Pose3, pose2: Pose3, **kwargs) -> bool:
        return pose1.position.y < pose2.position.y

    @staticmethod
    def behind_y(pose1: Pose3, pose2: Pose3, **kwargs) -> bool:
        return pose1.position.y > pose2.position.y

    @staticmethod
    def above(pose1: Pose3, pose2: Pose3, **kwargs) -> bool:
        return pose1.position.z > pose2.position.z

    @staticmethod
    def below(pose1: Pose3, pose2: Pose3, **kwargs) -> bool:
        return pose1.position.z < pose2.position.z

    @staticmethod
    def centered_along_x(pose1: Pose3, pose2: Pose3, **kwargs) -> bool:
        return abs(pose1.position.x - pose2.position.x) < 0.05

    @staticmethod
    def centered_along_y(pose1: Pose3, pose2: Pose3, **kwargs) -> bool:
        return abs(pose1.position.y - pose2.position.y) < 0.05

    @staticmethod
    def centered_along_z(pose1: Pose3, pose2: Pose3, **kwargs) -> bool:
        return abs(pose1.position.z - pose2.position.z) < 0.05

    @staticmethod
    def qualitative_distance(distance: float) -> int:
        if distance < 0.15:
            return PullingPredicateLibrary.SHORT_DISTANCE
        else:
            return PullingPredicateLibrary.LONG_DISTANCE

    @staticmethod
    def qualitative_time(t: float, **kwargs) -> int:
        if t < 1.5:
            return PullingPredicateLibrary.SHORT_DURATION
        else:
            return PullingPredicateLibrary.LONG_DURATION
