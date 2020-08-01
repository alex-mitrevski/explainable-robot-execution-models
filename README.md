# `explainable-robot-execution-models`

# Table of Contents

1. [Summary](#Summary)
2. [Setup](#Setup)
3. [Dependencies](#Dependencies)
4. [Learning Data](#Learning-Data)
5. [Execution Model Learning](#Execution-Model-Learning)
6. [Robot Experiments](#Robot-Experiments)
7. [Contributing](#Contributing)
8. [Notes](#Notes)

## Summary

This repository contains accompanying code for our IROS paper

`A. Mitrevski, P. G. Plöger, and G. Lakemeyer, "Representation and Experience-Based Learning of Explainable Models for Robot Action Execution," in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020.`

The repository only contains the code for learning execution models and sampling from them. See below for comments on using the learned models on a robot.

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

The learning data is stored in a MongoDB database in the format described in our [black-box](https://github.com/ropod-project/black-box) repository. The data processing code:
* depends on [`black-box-tools`](https://github.com/ropod-project/black-box-tools), which provides various utilities for working with MongoDB data
* uses various data structures defined in [`action-execution`](https://github.com/alex-mitrevski/action-execution)

## Learning Data

A subset of the recorded data, which is sufficient for learning the models that were used in the experiments in the paper, is provided [via Zenodo](https://zenodo.org/record/3968984).

The dataset contains three MongoDB databases - `handle_drawer_logs`, `handle_fridge_logs`, `pull_logs` - corresponding to the experiments for grasping a drawer handle, grasping a fridge handle, and object pulling, respectively.

## Execution Model Learning

The models that were used in our experiments can be learned after restoring the MongoDB databases. Learning scripts are provided under `scripts`.
* For learning the drawer handle execution model, the `drawer_handle_model_learner.py` can be used (this requires the `handle_drawer_logs` database to be present)
```
python3 scripts/drawer_handle_model_learner.py
```
* The fridge drawer handle execution model can be learned using the `fridge_handle_model_learner.py` script (the script requires the `handle_fridge_logs` database to be present)
```
python3 scripts/fridge_handle_model_learner.py
```
* The `pull_model_learner.py` script can be used for learning the object pulling model (the `pull_logs` database should be present)
```
python3 scripts/pull_model_learner.py
```

All three scripts save the learned model in pickle format.

## Robot Experiments

Our experiments were done using [`mas_domestic_robotics`](https://github.com/b-it-bots/mas_domestic_robotics), which contains configurable, robot-independent functionalities tailored at domestic robots. For the handle grasping experiment, we used the [handle opening action](https://github.com/b-it-bots/mas_domestic_robotics/tree/devel/mdr_planning/mdr_actions/mdr_manipulation_actions/mdr_handle_open_action). The [action for pushing/pulling objects](https://github.com/b-it-bots/mas_domestic_robotics/tree/devel/mdr_planning/mdr_actions/mdr_manipulation_actions/mdr_push_pull_object_action) was used for the object pulling experiment.

On the Toyota HSR, which we used for the experiments in the paper, we are still running Ubuntu 16.04 with ROS kinetic, which works best with Python 2.7. The code in this repository, on the other hand, is written for Python 3.5+ (specifically, we use type annotations quite extensively). For bridging the gap between the different versions, we used ROS-based "glue" scripts that were processing requests for execution parameters from the actions and were returning sampled execution parameters. These scripts are not included in this repository, but please get in touch with me if you would like to get access to them.

## Contributing

Contributions in the form of PRs (for example for adding new action models) as well as issue reports are welcome.

## Notes

* The code in this repository is, in a sense, branched off of [`action-execution`](https://github.com/alex-mitrevski/action-execution), so I may decide to merge the packages at some future point.
* Related to the previous point, this repository is ongoing work and will be updated over time. The version used for the IROS 2020 experiments is preserved by the [iros2020](https://github.com/alex-mitrevski/explainable-robot-execution-models/releases/tag/iros2020) tag.
