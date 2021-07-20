import numpy as np

from black_box_tools.db_utils import DBUtils
from black_box_tools.data_utils import DataUtils

from action_execution.extern import transformations as tf

from robot_action_execution.actions.grasp_handle import GraspHandleExecutionModel

def get_closest_handle(handles):
    distances = np.zeros(len(handles))
    for i, handle in enumerate(handles):
        distances[i] = np.linalg.norm([handle['pose']['pose']['position']['x'],
                                       handle['pose']['pose']['position']['y'],
                                       handle['pose']['pose']['position']['z']])
    return handles[np.argmin(distances)]

def get_handle_position(log_db_name, event_timestamp):
    object_detection_result = DBUtils.get_last_doc_before(log_db_name,
                                                          'ros_detect_handle_server_result',
                                                          event_timestamp)
    detected_handle = get_closest_handle(object_detection_result['result']['objects']['objects'])
    detected_handle_bb = detected_handle['bounding_box']
    center_position = [detected_handle_bb['center']['x'],
                       detected_handle_bb['center']['y'],
                       detected_handle_bb['center']['z']]
    return np.array(center_position)

def get_handle_bbox(log_db_name, event_timestamp):
    object_detection_result = DBUtils.get_last_doc_before(log_db_name,
                                                          'ros_detect_handle_server_result',
                                                          event_timestamp)
    detected_handle = get_closest_handle(object_detection_result['result']['objects']['objects'])
    detected_handle_bb = detected_handle['bounding_box']

    bb_min = [detected_handle_bb['center']['x'] - (detected_handle_bb['dimensions']['x'] / 2.),
              detected_handle_bb['center']['y'] - (detected_handle_bb['dimensions']['y'] / 2.),
              detected_handle_bb['center']['z'] - (detected_handle_bb['dimensions']['z'] / 2.)]

    bb_max = [detected_handle_bb['center']['x'] + (detected_handle_bb['dimensions']['x'] / 2.),
              detected_handle_bb['center']['y'] + (detected_handle_bb['dimensions']['y'] / 2.),
              detected_handle_bb['center']['z'] + (detected_handle_bb['dimensions']['z'] / 2.)]

    return bb_min, bb_max

def get_goal_pose(log_db_name, event_timestamp):
    goal_pose_doc = DBUtils.get_last_doc_before(log_db_name,
                                                'ros_grasped_handle_pose',
                                                 event_timestamp)

    goal_position = [goal_pose_doc['pose']['position']['x'],
                     goal_pose_doc['pose']['position']['y'],
                     goal_pose_doc['pose']['position']['z']]

    goal_orientation_q = [goal_pose_doc['pose']['orientation']['x'],
                          goal_pose_doc['pose']['orientation']['y'],
                          goal_pose_doc['pose']['orientation']['z'],
                          goal_pose_doc['pose']['orientation']['w']]

    orientation_euler = tf.euler_from_quaternion(goal_orientation_q)
    return np.array(goal_position), np.array(orientation_euler)

def get_label(event_start, event_end):
    if event_start and event_end:
        if event_start == 'success':
            return 1
        else:
            return 0
    return 0

if __name__ == '__main__':
    log_db_name = 'handle_fridge_logs'

    event_docs = DBUtils.get_all_docs(log_db_name, 'ros_event')
    event_timestamps = DataUtils.get_all_measurements(event_docs, 'timestamp')
    event_descriptions = DataUtils.get_all_measurements(event_docs, 'description')

    data_point_count = len(event_timestamps) // 2

    handle_positions = np.zeros((data_point_count, 3))
    handle_bb_mins = np.zeros((data_point_count, 3))
    handle_bb_maxs = np.zeros((data_point_count, 3))

    goal_positions = np.zeros((data_point_count, 3))
    goal_orientations = np.zeros((data_point_count, 3))
    labels = np.zeros(data_point_count, dtype=int)
    data_idx = 0
    for i in range(0, data_point_count*2, 2):
        handle_positions[data_idx] = get_handle_position(log_db_name, event_timestamps[i])
        handle_bb_mins[data_idx], handle_bb_maxs[data_idx] = get_handle_bbox(log_db_name, event_timestamps[i])
        goal_positions[data_idx], goal_orientations[data_idx] = get_goal_pose(log_db_name, event_timestamps[i])
        labels[data_idx] = get_label(event_descriptions[i], event_descriptions[i+1])
        data_idx += 1

    action_parameters = handle_positions - goal_positions
    execution_model = GraspHandleExecutionModel('fridge-handle-grasp', 'FridgeHandle')

    print('Learning preconditions...')
    execution_model.learn_preconditions(data_point_count, goal_positions,
                                        goal_orientations, handle_bb_mins,
                                        handle_bb_maxs, labels)

    print('Learning success model...')
    execution_model.learn_success_model(action_parameters, labels)
    execution_model.save('fridge-handle-grasp-execution-model.pkl')
