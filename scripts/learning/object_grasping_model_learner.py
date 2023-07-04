import argparse
import numpy as np

from black_box_tools.db_utils import DBUtils
from black_box_tools.data_utils import DataUtils

from action_execution.extern import transformations as tf

from robot_action_execution.actions.grasp_object import GraspObjectExecutionModel

def get_object_pose(log_db_name, event_timestamp):
    grasped_object = DBUtils.get_last_doc_before(log_db_name,
                                                 'ros_grasped_object',
                                                 event_timestamp)
    object_position = [grasped_object['pose']['pose']['position']['x'],
                       grasped_object['pose']['pose']['position']['y'],
                       grasped_object['pose']['pose']['position']['z']]

    object_orientation_q = [grasped_object['pose']['pose']['orientation']['w'],
                            grasped_object['pose']['pose']['orientation']['x'],
                            grasped_object['pose']['pose']['orientation']['y'],
                            grasped_object['pose']['pose']['orientation']['z']]
    orientation_euler = tf.euler_from_quaternion(object_orientation_q)
    return np.array(object_position), np.array(orientation_euler)

def get_object_bbox(log_db_name, event_timestamp):
    grasped_object = DBUtils.get_last_doc_before(log_db_name,
                                                 'ros_grasped_object',
                                                 event_timestamp)
    grasped_object_bb = grasped_object['bounding_box']

    bb_min = [grasped_object_bb['center']['x'] - (grasped_object_bb['dimensions']['x'] / 2.),
              grasped_object_bb['center']['y'] - (grasped_object_bb['dimensions']['y'] / 2.),
              grasped_object_bb['center']['z'] - (grasped_object_bb['dimensions']['z'] / 2.)]

    bb_max = [grasped_object_bb['center']['x'] + (grasped_object_bb['dimensions']['x'] / 2.),
              grasped_object_bb['center']['y'] + (grasped_object_bb['dimensions']['y'] / 2.),
              grasped_object_bb['center']['z'] + (grasped_object_bb['dimensions']['z'] / 2.)]

    return bb_min, bb_max

def get_goal_pose(log_db_name, event_timestamp):
    goal_doc = DBUtils.get_last_doc_before(log_db_name,
                                           'ros_pickup_server_goal',
                                           event_timestamp)

    goal_position = [goal_doc['goal']['pose']['pose']['position']['x'],
                     goal_doc['goal']['pose']['pose']['position']['y'],
                     goal_doc['goal']['pose']['pose']['position']['z']]

    goal_orientation_q = [goal_doc['goal']['pose']['pose']['orientation']['w'],
                          goal_doc['goal']['pose']['pose']['orientation']['x'],
                          goal_doc['goal']['pose']['pose']['orientation']['y'],
                          goal_doc['goal']['pose']['pose']['orientation']['z']]

    orientation_euler = tf.euler_from_quaternion(goal_orientation_q)
    return np.array(goal_position), np.array(orientation_euler)

def get_grasp_strategy(log_db_name, event_timestamp):
    goal_doc = DBUtils.get_last_doc_before(log_db_name,
                                           'ros_pickup_server_goal',
                                           event_timestamp)
    return int(goal_doc['goal']['strategy'])

def get_label(event_start, event_end):
    if event_start and event_end:
        if event_start == 'success':
            return 1
        else:
            return 0
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='USAGE: python3 object_grasping_model_learner.py -on object-name -oc object-category')
    parser.add_argument('-on', '--object-name', type=str, required=True,
                        help='Name of the object whose model should be learned')
    parser.add_argument('-oc', '--object-category', type=str, required=True,
                        help='Category of the object (e.g. as specified in an ontology)')
    args = parser.parse_args()
    log_db_name = f'{args.object_name}_grasping_logs'

    event_docs = DBUtils.get_all_docs(log_db_name, 'ros_event')
    event_timestamps = DataUtils.get_all_measurements(event_docs, 'timestamp')
    event_descriptions = DataUtils.get_all_measurements(event_docs, 'description')

    data_point_count = len(event_timestamps) // 2

    object_positions = np.zeros((data_point_count, 3))
    object_orientations = np.zeros((data_point_count, 3))
    object_bb_mins = np.zeros((data_point_count, 3))
    object_bb_maxs = np.zeros((data_point_count, 3))

    goal_positions = np.zeros((data_point_count, 3))
    goal_orientations = np.zeros((data_point_count, 3))
    grasp_strategies = np.zeros(data_point_count, dtype=int) # currently unused
    labels = np.zeros(data_point_count, dtype=int)
    data_idx = 0
    for i in range(0, data_point_count*2, 2):
        object_positions[data_idx], object_orientations[data_idx] = get_object_pose(log_db_name, event_timestamps[i])
        object_bb_mins[data_idx], object_bb_maxs[data_idx] = get_object_bbox(log_db_name, event_timestamps[i])
        goal_positions[data_idx], goal_orientations[data_idx] = get_goal_pose(log_db_name, event_timestamps[i])
        grasp_strategies[data_idx] = get_grasp_strategy(log_db_name, event_timestamps[i])
        labels[data_idx] = get_label(event_descriptions[i], event_descriptions[i+1])
        data_idx += 1

    # normalising the positions with respect to the bounding box sizes
    # and calculating the relative gripper-object orientations
    # for the success prediction model
    relative_goal_positions = goal_positions - object_positions

    object_half_sizes = (object_bb_maxs - object_bb_mins) / 2.
    normalised_relative_goal_positions = relative_goal_positions / object_half_sizes

    relative_orientations = goal_orientations - object_orientations
    relative_orientations[np.where(relative_orientations>np.pi/2)] -= np.pi
    relative_orientations[np.where(relative_orientations<-np.pi/2)] += np.pi

    action_parameters = np.hstack((normalised_relative_goal_positions,
                                   relative_orientations[:,2][np.newaxis].T))

    execution_model = GraspObjectExecutionModel(f'{args.object_name}-grasp', args.object_category)

    print('Learning preconditions...')
    execution_model.learn_preconditions(data_point_count, goal_positions,
                                        goal_orientations, object_positions,
                                        object_orientations, object_bb_mins,
                                        object_bb_maxs, labels)

    print('Learning success model...')
    execution_model.learn_success_model(action_parameters, labels)

    model_path = f'{args.object_name}-grasp-execution-model.pkl'
    print(f'Saving model to {model_path}...')
    execution_model.save(model_path)
