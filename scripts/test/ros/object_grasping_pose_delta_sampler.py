#!/usr/bin/env python3
import os
from copy import deepcopy

import rospy

from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
from mas_perception_msgs.msg import Object

from mas_knowledge_utils.ontology_query_interface import OntologyQueryInterface
from mas_tools.ros_utils import get_package_path

from action_execution.geometry.vector import Vector3
from action_execution.geometry.pose import Pose3
from action_execution.geometry.bbox import BBox3
import action_execution.extern.transformations as tf

from robot_action_execution.predicates.grasping import GraspingPredicateLibrary
from robot_action_execution.action_execution_model_utils import ActionExecutionModelUtils


def state_update_fn(state_update, pose, b_box, object_pose):
    p_copy = deepcopy(pose)

    object_half_size_x = (b_box.max.x - b_box.min.x) / 2.
    object_half_size_y = (b_box.max.y - b_box.min.y) / 2.
    object_half_size_z = (b_box.max.z - b_box.min.z) / 2.

    absolute_state_update_x = state_update[0] * object_half_size_x
    absolute_state_update_y = state_update[1] * object_half_size_y
    absolute_state_update_z = state_update[2] * object_half_size_z
    orientation_update = state_update[3]

    p_copy.position.x += absolute_state_update_x
    p_copy.position.y += absolute_state_update_y
    p_copy.position.z += absolute_state_update_z
    p_copy.orientation.z += orientation_update
    return {'pose': p_copy, 'b_box': b_box, 'object_pose': object_pose}

def get_object_pose(object_msg):
    object_position = Vector3(x=object_msg.pose.pose.position.x,
                              y=object_msg.pose.pose.position.y,
                              z=object_msg.pose.pose.position.z)
    euler_orientation = tf.euler_from_quaternion([object_msg.pose.pose.orientation.w,
                                                  object_msg.pose.pose.orientation.x,
                                                  object_msg.pose.pose.orientation.y,
                                                  object_msg.pose.pose.orientation.z])
    object_orientation = Vector3(x=euler_orientation[0],
                                 y=euler_orientation[1],
                                 z=euler_orientation[2])
    object_pose = Pose3(position=object_position,
                        orientation=object_orientation)
    return object_pose

def get_object_bbox(object_msg):
    bbox_msg = object_msg.bounding_box
    bb_min = Vector3(x=bbox_msg.center.x - (bbox_msg.dimensions.x / 2.),
                     y=bbox_msg.center.y - (bbox_msg.dimensions.y / 2.),
                     z=bbox_msg.center.z - (bbox_msg.dimensions.z / 2.))
    bb_max = Vector3(x=bbox_msg.center.x + (bbox_msg.dimensions.x / 2.),
                     y=bbox_msg.center.y + (bbox_msg.dimensions.y / 2.),
                     z=bbox_msg.center.z + (bbox_msg.dimensions.z / 2.))
    object_bbox = BBox3(min_values=bb_min, max_values=bb_max)
    return object_bbox

def get_absolute_delta(state_update, b_box):
    object_half_size_x = (b_box.max.x - b_box.min.x) / 2.
    object_half_size_y = (b_box.max.y - b_box.min.y) / 2.
    object_half_size_z = (b_box.max.z - b_box.min.z) / 2.

    absolute_state_update_x = state_update[0] * object_half_size_x
    absolute_state_update_y = state_update[1] * object_half_size_y
    absolute_state_update_z = state_update[2] * object_half_size_z
    orientation_update = state_update[3]

    return [absolute_state_update_x, absolute_state_update_y,
            absolute_state_update_z, orientation_update]

class PoseSmartPub(object):
    YCB_ANNOTATION_TO_ONTOLOGY_MAP = {
        '001_chips_can': 'ChipsCan',
        '003_cracker_box': 'CrackerBox',
        '004_sugar_box': 'SugarBox',
        '005_tomato_soup_can': 'TomatoCan',
        '006_mustard_bottle': 'MustardContainer',
        '011_banana': 'Banana',
        '012_strawberry': 'Strawberry',
        '013_apple': 'Apple',
        '017_orange': 'Orange',
        '019_pitcher_base': 'Pitcher',
        '023_wine_glass': 'WineGlass',
        '025_mug': 'Mug',
        '055_baseball': 'Baseball',
        '056_tennis_ball': 'TennisBall',
        '057_racquetball': 'Racquetball'
    }

    def __init__(self):
        self.delta = None
        self.manipulated_obj = None
        self.selected_obj = None
        ontology_file_url = os.path.join('file://{0}'.format(get_package_path('mas_knowledge_base')),
                                         'common/ontology/apartment.owl')
        ontology_base_url = 'http://apartment.owl'
        ontology_entity_delimiter = '#'
        ontology_ns = ''
        self.ontology_interface = OntologyQueryInterface(ontology_file=ontology_file_url,
                                                         base_url=ontology_base_url,
                                                         entity_delimiter=ontology_entity_delimiter,
                                                         class_prefix=ontology_ns)

        self.delta_pub = rospy.Publisher('/pose_delta', Pose, queue_size=1)
        self.selected_obj_pub = rospy.Publisher('/generalising_object', String, queue_size=1)
        self.request_sub = rospy.Subscriber('/get_pose_delta', Object, self.get_delta)
        self.result_sub = rospy.Subscriber('/success', Bool, self.update_generalisation_model)

    def get_delta(self, object_msg):
        category = PoseSmartPub.YCB_ANNOTATION_TO_ONTOLOGY_MAP[object_msg.category]
        self.manipulated_obj = category
        rospy.loginfo('Retrieving execution data for %s', category)

        grasp_execution_model = ActionExecutionModelUtils.get_execution_model(action_name='Grasp',
                                                                              obj_type=category,
                                                                              ontology_interface=self.ontology_interface)

        object_pose = get_object_pose(object_msg)
        gripper_pose = get_object_pose(object_msg)
        object_bbox = get_object_bbox(object_msg)

        self.selected_obj = grasp_execution_model.object_type
        delta = grasp_execution_model.sample_state(value=1,
                                                   predicate_library=GraspingPredicateLibrary,
                                                   state_update_fn=state_update_fn,
                                                   pose=gripper_pose, b_box=object_bbox,
                                                   object_pose=object_pose)
        self.delta = get_absolute_delta(delta, object_bbox)

    def update_generalisation_model(self, bool_msg):
        rospy.loginfo('Updating model of %s', self.manipulated_obj)
        ActionExecutionModelUtils.update_model_generalisation_attempts(action_name='Grasp',
                                                                       obj_generalised_to=self.manipulated_obj,
                                                                       generalising_obj=self.selected_obj,
                                                                       execution_success=bool_msg.data)

if __name__ == '__main__':
    rospy.init_node('grasping_position_delta_sampler')
    pose_smart_pub = PoseSmartPub()

    rospy.loginfo('Waiting for requests on /get_pose_delta and publishing responses to /pose_delta')
    pose_delta_msg = Pose()
    while not rospy.is_shutdown():
        if pose_smart_pub.delta is not None:
            print('Delta: {0}'.format(pose_smart_pub.delta))

            pose_delta_msg.position.x = pose_smart_pub.delta[0]
            pose_delta_msg.position.y = pose_smart_pub.delta[1]
            pose_delta_msg.position.z = pose_smart_pub.delta[2]

            quaternion_orientation = tf.quaternion_from_euler(0., 0., pose_smart_pub.delta[3])
            pose_delta_msg.orientation.w = quaternion_orientation[0]
            pose_delta_msg.orientation.x = quaternion_orientation[1]
            pose_delta_msg.orientation.y = quaternion_orientation[2]
            pose_delta_msg.orientation.z = quaternion_orientation[3]

            pose_smart_pub.delta_pub.publish(pose_delta_msg)
            pose_smart_pub.selected_obj_pub.publish(pose_smart_pub.selected_obj)
            pose_smart_pub.delta = None
        rospy.sleep(0.1)
