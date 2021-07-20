# `explainable-robot-execution-models`

# Table of Contents

1. [Summary](#Summary)
2. [Setup](#Setup)
3. [Dependencies](#Dependencies)
4. [IROS 2020 Paper](#IROS-2020-Paper)
    1. [Learning Data](#Learning-Data)
    2. [Execution Model Learning](#Execution-Model-Learning)
5. [ICRA 2021 Paper](#ICRA-2021-Paper)
    1. [Data](#Data)
    2. [Execution Failure Diagnosis and Experience Correction](#Execution-Failure-Diagnosis-and-Experience-Correction)
6. [IROS 2021 Paper](#IROS-2021-Paper)
    1. [Data](#Data)
    2. [Model Genaralisation Example](#Model-Genaralisation-Example)
7. [Robot Experiments](#Robot-Experiments)
8. [Contributing](#Contributing)
9. [Notes](#Notes)

## Summary

This repository contains accompanying code for the following papers:
* `A. Mitrevski, P. G. Plöger, and G. Lakemeyer, "Ontology-Assisted Generalisation of Robot Action Execution Knowledge," in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021.`
* `A. Mitrevski, P. G. Plöger, and G. Lakemeyer, "Robot Action Diagnosis and Experience Correction by Falsifying Parameterised Execution Models," in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2021.`
* `A. Mitrevski, P. G. Plöger, and G. Lakemeyer, "Representation and Experience-Based Learning of Explainable Models for Robot Action Execution," in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5641-5647, 2020. Winner of the "Best Paper Award on Cognitive Robotics".`

See below for comments on using execution models on a robot.

## Setup

1. Clone this repository
```
git clone https://github.com/alex-mitrevski/explainable-robot-execution-models.git
```
2. Install the dependencies
```
cd explainable-robot-execution-models
pip3 install -r requirements.txt
```
3. Set up the package:
```
python3 setup.py [install|develop]
```

## Dependencies

The code in this repository uses `numpy` and `scikit-learn` (specifically, the scikit-learn implementations of Gaussian Process regression and k-nearest neighbours).

The data for learning execution models is stored in a MongoDB database in the format described in our [black-box](https://github.com/ropod-project/black-box) repository. The data processing code:
* depends on [`black-box-tools`](https://github.com/ropod-project/black-box-tools), which provides various utilities for working with MongoDB data
* uses various data structures defined in [`action-execution`](https://github.com/alex-mitrevski/action-execution)

The ontology-based model generalisation code additionally depends on the [`mas_knowledge_base`](https://github.com/b-it-bots/mas_knowledge_base) package.

Some of the test scripts depend on utility functions defined in [`mas_tools`](https://github.com/b-it-bots/mas_tools) as well as on the ROS messages defined in [`mas_perception_msgs`](https://github.com/b-it-bots/mas_perception_msgs).

## IROS 2020 Paper

### Learning Data

A subset of the recorded data, which is sufficient for learning the models that were used in the experiments in the IROS 2020 paper, is provided [via Zenodo](https://zenodo.org/record/3968984).

The dataset contains three MongoDB databases - `handle_drawer_logs`, `handle_fridge_logs`, `pull_logs` - corresponding to the experiments for grasping a drawer handle, grasping a fridge handle, and object pulling, respectively.

### Execution Model Learning

The models that were used in our experiments can be learned after restoring the MongoDB databases. Learning scripts are provided under `scripts/learning`.
* For learning the drawer handle execution model, the `drawer_handle_model_learner.py` can be used (this requires the `handle_drawer_logs` database to be present)
```
python3 scripts/learning/drawer_handle_model_learner.py
```
* The fridge drawer handle execution model can be learned using the `fridge_handle_model_learner.py` script (the script requires the `handle_fridge_logs` database to be present)
```
python3 scripts/learning/fridge_handle_model_learner.py
```
* The `pull_model_learner.py` script can be used for learning the object pulling model (the `pull_logs` database should be present)
```
python3 scripts/learning/pull_model_learner.py
```

All three scripts save the learned model in pickle format.

## ICRA 2021 Paper

### Data

The data used in the ICRA 2021 paper is also available [via Zenodo](https://zenodo.org/record/4603348). The provided dataset includes a MongoDB database (`drawer_handle_grasping_failures`), which includes data from which execution parameters can be extracted as well as ground-truth failure annotations. Additionally, images from our robot's hand before moving the arm towards the handle and just before closing the gripper are also included in the dataset.

### Execution Failure Diagnosis and Experience Correction

A script that illustrates the diagnosis is provided under `scripts/test`:

```
python3 scripts/test/experience_correction.py
```

This script expects the `drawer_handle_grasping_failures` database to be present, so it is necessary to download the dataset and restore the database first. The script diagnoses the failures in the dataset and suggests corrective execution parameters whenever possible; for illustrative purposes, the original and corrected execution parameters are plotted after parameter correction.

## IROS 2021 Paper

### Data

The data from the experiments in the IROS 2021 paper can also be found [on Zenodo](https://zenodo.org/record/4551725). The dataset includes both learning and generalisation data for all objects used in the experiments and for both actions (grasping and stowing). The grasping dataset also includes images from the robot's hand camera (i) just before approaching the object and (ii) just before grasping.

### Model Genaralisation Example

A test script that illustrates how the model generalisation works during execution of an object grasping action is provided under `scripts/test/ros`:

```
python3 scripts/test/ros/object_grasping_pose_delta_sampler.py
```

1. This script creates a ROS node that subscribes to the topic `/get_pose_delta` on which an `mas_perception_msgs/Object` message is published; this includes information about an object to be grasped (such as object type, estimated pose, and bounding box size).
2. When an object is received, an execution model is selected and action parameters are sampled from the model; in the grasping case, these parameters represent (i) a position offset from the center of the object's bounding box and (ii) a relative wrist orientation (along the xy-plane) with respect to the object's estimated orientation. Once the parameters are sampled, the node publishes:
    1. a `geometry_msgs/Pose` message with the sampled parameters and
    2. an `std_msgs/String` message with the object type whose model was used for executing the action
3. The node also subscribes to the `/success` topic, which is of type `std_msgs.Bool`. The intended use here is that the outcome of the execution is published on this topic (success or failure); when a message is received, the data about the attempted model generalisations are updated (these are stored in a MongoDB database).

## Robot Experiments

Our experiments were done using [`mas_domestic_robotics`](https://github.com/b-it-bots/mas_domestic_robotics), which contains configurable, robot-independent functionalities tailored at domestic robots. For the handle grasping experiment, we used the [handle opening action](https://github.com/b-it-bots/mas_domestic_robotics/tree/devel/mdr_planning/mdr_actions/mdr_manipulation_actions/mdr_handle_open_action). The [action for pushing/pulling objects](https://github.com/b-it-bots/mas_domestic_robotics/tree/devel/mdr_planning/mdr_actions/mdr_manipulation_actions/mdr_push_pull_object_action) was used for the object pulling experiment. The [pickup action](https://github.com/b-it-bots/mas_domestic_robotics/tree/devel/mdr_planning/mdr_actions/mdr_manipulation_actions/mdr_pickup_action) was used for the object grasping experiments, while the [throwing action](https://github.com/b-it-bots/mas_domestic_robotics/tree/devel/mdr_planning/mdr_actions/mdr_manipulation_actions/mdr_place_action) (which is an instance of the placing action) was used for the object stowing experiments.

On the Toyota HSR, which we used for the experiments in the paper, we are still running Ubuntu 16.04 with ROS kinetic, which works best with Python 2.7. The code in this repository, on the other hand, is written for Python 3.5+ (specifically, we use type annotations quite extensively). For bridging the gap between the different versions, we used ROS-based "glue" scripts that were processing requests for execution parameters from the actions and were returning sampled execution parameters. These scripts are not included in this repository, but please get in touch with me if you would like to get access to them.

## Contributing

Contributions in the form of PRs (for example for adding new action models) as well as issue reports are welcome.

## Notes

* The code in this repository is, in a sense, branched off of [`action-execution`](https://github.com/alex-mitrevski/action-execution), so I may decide to merge the packages at some future point.
* Related to the previous point, this repository is ongoing work and will be updated over time, but:
    * the version used for the IROS 2020 experiments is preserved by the [iros2020](https://github.com/alex-mitrevski/explainable-robot-execution-models/releases/tag/iros2020) tag.
    * the version used for the ICRA 2021 experiments is preserved by the [icra2021](https://github.com/alex-mitrevski/explainable-robot-execution-models/releases/tag/icra2021) tag.
    * the version used for the IROS 2021 experiments is preserved by the [iros2021](https://github.com/alex-mitrevski/explainable-robot-execution-models/releases/tag/iros2021) tag.
* The component for learning relational preconditions is reused from [my old implementation](https://github.com/alex-mitrevski/delta-execution-models/blob/master/rule_learner/symblearn/stat_learn.py), but includes some bug fixes.
