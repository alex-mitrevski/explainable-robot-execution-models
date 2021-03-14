from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from action_execution.geometry.vector import Vector3
from action_execution.geometry.pose import Pose3
from action_execution.geometry.bbox import BBox3

from black_box_tools.db_utils import DBUtils
from black_box_tools.data_utils import DataUtils

from action_execution.extern import transformations as tf

from robot_action_execution.actions.grasp_handle import GraspHandleExecutionModel
from robot_action_execution.action_execution_model_utils import FailureSearchParams
from robot_action_execution.predicates.grasping import GraspingPredicateLibrary

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

def state_update_fn(state_update, pose, b_box):
    p_copy = deepcopy(pose)
    p_copy.position.x += state_update[0]
    p_copy.position.y += state_update[1]
    p_copy.position.z += state_update[2]
    return {'pose': p_copy, 'b_box': b_box}

if __name__ == '__main__':
    log_db_name = 'drawer_handle_grasping_failures'

    event_docs = DBUtils.get_all_docs(log_db_name, 'ros_event')
    event_timestamps = DataUtils.get_all_measurements(event_docs, 'timestamp')
    event_descriptions = DataUtils.get_all_measurements(event_docs, 'description')

    ground_truth_diagnoses = DBUtils.get_all_docs(log_db_name, 'fault_diagnoses_ground_truth')

    data_point_count = len(event_timestamps) // 2
    handle_positions = np.zeros((data_point_count, 3))
    handle_bb_mins = np.zeros((data_point_count, 3))
    handle_bb_maxs = np.zeros((data_point_count, 3))
    goal_positions = np.zeros((data_point_count, 3))
    goal_orientations = np.zeros((data_point_count, 3))
    labels = np.zeros(data_point_count, dtype=int)
    data_idx = 0

    print('Extracting data...')
    for i in range(0, data_point_count*2, 2):
        handle_positions[data_idx] = get_handle_position(log_db_name, event_timestamps[i])
        handle_bb_mins[data_idx], handle_bb_maxs[data_idx] = get_handle_bbox(log_db_name, event_timestamps[i])

        goal_positions[data_idx], goal_orientations[data_idx] = get_goal_pose(log_db_name, event_timestamps[i])

        labels[data_idx] = get_label(event_descriptions[i], event_descriptions[i+1])
        data_idx += 1

    action_parameters = handle_positions - goal_positions
    execution_model = GraspHandleExecutionModel('drawer-handle-grasp')

    print('Setting preconditions')
    execution_model.preconditions = preconditions = [('in_front_of_x', ['gripper', 'x'], 1),
                                                     ('far_in_front_of_x', ['gripper', 'x'], 0),
                                                     ('behind_x', ['gripper', 'x'], 0),
                                                     ('in_front_of_y', ['gripper', 'x'], 0),
                                                     ('behind_y', ['gripper', 'x'], 0),
                                                     ('above', ['gripper', 'x'], 0),
                                                     ('below', ['gripper', 'x'], 0),
                                                     ('centered_along_x', ['gripper', 'x'], 0),
                                                     ('centered_along_z', ['gripper', 'x'], 1)]

    print('Learning success model...')
    execution_model.learn_success_model(action_parameters, labels)

    execution_model.parameter_idx_mappings = {'pose.position.x': 0,
                                              'pose.position.y': 1,
                                              'pose.position.z': 2}

    handle_size_x = np.mean(handle_bb_maxs[:,0] - handle_bb_mins[:,0])
    handle_size_y = np.mean(handle_bb_maxs[:,1] - handle_bb_mins[:,1])
    handle_size_z = np.mean(handle_bb_maxs[:,2] - handle_bb_mins[:,2])
    failure_search_params = FailureSearchParams(parameter_stds=np.array([handle_size_x/10,
                                                                         handle_size_y/10,
                                                                         handle_size_z/10]),
                                                max_sample_count=200,
                                                range_increase_percentage=5)

    print('Diagnosing failed executions and suggesting corrections...')
    original_goal_positions = []
    alternative_experiences = []
    for i in range(goal_positions.shape[0]):
        if labels[i] == 1: continue

        gripper_position = Vector3(x=goal_positions[i,0],
                                   y=goal_positions[i,1],
                                   z=goal_positions[i,2])
        gripper_orientation = Vector3(x=goal_orientations[i,0],
                                      y=goal_orientations[i,1],
                                      z=goal_orientations[i,2])

        gripper_pose = Pose3(position=gripper_position,
                             orientation=gripper_orientation)

        bb_min = Vector3(x=handle_bb_mins[i,0],
                         y=handle_bb_mins[i,1],
                         z=handle_bb_mins[i,2])
        bb_max = Vector3(x=handle_bb_maxs[i,0],
                         y=handle_bb_maxs[i,1],
                         z=handle_bb_maxs[i,2])

        handle_bbox = BBox3(min_values=bb_min,
                            max_values=bb_max)

        diagnoses, experience_correction = execution_model.correct_failed_experience(predicate_library=GraspingPredicateLibrary,
                                                                                     state_update_fn=state_update_fn,
                                                                                     failure_search_params=failure_search_params,
                                                                                     diagnosis_repetition_count=50,
                                                                                     diagnosis_candidate_confidence=0.8,
                                                                                     correction_sample_count=10,
                                                                                     pose=gripper_pose,
                                                                                     b_box=handle_bbox)
        alternative_experience = action_parameters[i] + experience_correction
        if not np.allclose(experience_correction, np.zeros_like(experience_correction)):
            original_goal_positions.append(action_parameters[i])
            alternative_experiences.append(alternative_experience)

        print('Sample: {0}'.format(i+1))
        print('Relative position: {0}'.format(action_parameters[i]))
        print('Diagnosis candidates: {0}'.format(diagnoses))
        print('Experience correction: {0}'.format(experience_correction))
        print('Alternative experience: {0}'.format(alternative_experience))
        print()

    original_goal_positions = np.array(original_goal_positions)
    alternative_experiences = np.array(alternative_experiences)
    print('Number of corrected experiences: {0}'.format(len(alternative_experiences)))

    print('Plotting failed executions and corrections...')
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(original_goal_positions[:,0], original_goal_positions[:,1], original_goal_positions[:,2], color='r')
    ax.scatter(alternative_experiences[:,0], alternative_experiences[:,1], alternative_experiences[:,2], color='g')
    ax.quiver(original_goal_positions[:,0], original_goal_positions[:,1], original_goal_positions[:,2],
              alternative_experiences[:,0] - original_goal_positions[:,0],
              alternative_experiences[:,1] - original_goal_positions[:,1],
              alternative_experiences[:,2] - original_goal_positions[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(np.min(original_goal_positions[:,0]), np.max(original_goal_positions[:,0]))
    ax.set_ylim(np.min(original_goal_positions[:,1]), np.max(original_goal_positions[:,1]))
    ax.set_zlim(np.min(original_goal_positions[:,2]), np.max(original_goal_positions[:,2]))

    ax.view_init(10, 180)
    plt.show()
