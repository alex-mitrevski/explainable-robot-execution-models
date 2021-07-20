import numpy as np

from black_box_tools.db_utils import DBUtils
from black_box_tools.data_utils import DataUtils

from action_execution.extern import transformations as tf

from robot_action_execution.actions.pull import PullExecutionModel

def get_object_pose(log_db_name, event_timestamp):
    goal_pose_doc = DBUtils.get_last_doc_before(log_db_name,
                                                'ros_push_pull_object_server_goal',
                                                 event_timestamp)

    goal_position = [goal_pose_doc['goal']['object_pose']['pose']['position']['x'],
                     goal_pose_doc['goal']['object_pose']['pose']['position']['y'],
                     goal_pose_doc['goal']['object_pose']['pose']['position']['z']]

    goal_orientation_q = [goal_pose_doc['goal']['object_pose']['pose']['orientation']['x'],
                          goal_pose_doc['goal']['object_pose']['pose']['orientation']['y'],
                          goal_pose_doc['goal']['object_pose']['pose']['orientation']['z'],
                          goal_pose_doc['goal']['object_pose']['pose']['orientation']['w']]

    orientation_euler = tf.euler_from_quaternion(goal_orientation_q)
    return np.array(goal_position), np.array(orientation_euler)

def get_goal_pose(log_db_name, event_timestamp):
    goal_pose_doc = DBUtils.get_last_doc_before(log_db_name,
                                                'ros_push_pull_object_server_goal',
                                                 event_timestamp)

    goal_position = [goal_pose_doc['goal']['goal_pose']['pose']['position']['x'],
                     goal_pose_doc['goal']['goal_pose']['pose']['position']['y'],
                     goal_pose_doc['goal']['goal_pose']['pose']['position']['z']]

    goal_orientation_q = [goal_pose_doc['goal']['goal_pose']['pose']['orientation']['x'],
                          goal_pose_doc['goal']['goal_pose']['pose']['orientation']['y'],
                          goal_pose_doc['goal']['goal_pose']['pose']['orientation']['z'],
                          goal_pose_doc['goal']['goal_pose']['pose']['orientation']['w']]

    orientation_euler = tf.euler_from_quaternion(goal_orientation_q)
    return np.array(goal_position), np.array(orientation_euler)

def get_distance_from_edge(log_db_name, event_timestamp):
    distance_doc = DBUtils.get_last_doc_before(log_db_name,
                                               'ros_distance_from_edge',
                                                event_timestamp)
    return distance_doc['data']

def get_motion_duration(log_db_name, event_timestamp):
    duration_doc = DBUtils.get_last_doc_before(log_db_name,
                                               'ros_motion_duration',
                                                event_timestamp)
    return duration_doc['data']

def get_label(event_start, event_end):
    if event_start and event_end:
        if event_start == 'success':
            return 1
        else:
            return 0
    return 0

if __name__ == '__main__':
    log_db_name = 'pull_logs'

    event_docs = DBUtils.get_all_docs(log_db_name, 'ros_event')
    event_timestamps = DataUtils.get_all_measurements(event_docs, 'timestamp')
    event_descriptions = DataUtils.get_all_measurements(event_docs, 'description')

    data_point_count = len(event_timestamps) // 2
    object_positions = np.zeros((data_point_count, 3))
    goal_positions = np.zeros((data_point_count, 3))
    distances_from_edge = np.zeros((data_point_count, 1))
    motion_durations = np.zeros((data_point_count, 1))

    labels = np.zeros(data_point_count, dtype=int)
    data_idx = 0
    for i in range(0, data_point_count*2, 2):
        object_positions[data_idx], _ = get_object_pose(log_db_name, event_timestamps[i])
        goal_positions[data_idx], _ = get_goal_pose(log_db_name, event_timestamps[i])
        distances_from_edge[data_idx] = get_distance_from_edge(log_db_name, event_timestamps[i])
        motion_durations[data_idx] = get_motion_duration(log_db_name, event_timestamps[i])
        labels[data_idx] = get_label(event_descriptions[i], event_descriptions[i+1])
        data_idx += 1

    relative_goal_positions = object_positions - goal_positions
    action_parameters = np.hstack((relative_goal_positions,
                                   distances_from_edge,
                                   motion_durations))

    execution_model = PullExecutionModel('pull', 'YogurtCup')

    print('Learning preconditions...')
    execution_model.learn_preconditions(data_point_count, object_positions,
                                        goal_positions, motion_durations,
                                        labels)

    print('Learning success model...')
    execution_model.learn_success_model(action_parameters, labels)
    execution_model.save('pull-execution-model.pkl')
