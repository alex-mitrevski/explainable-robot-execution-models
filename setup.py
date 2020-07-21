from setuptools import setup, find_packages

setup(name='robot_action_execution',
      version='0.0.1',
      description='Library and utilities for learning robot action execution models',
      url='https://github.com/alex-mitrevski/explainable-robot-execution-models',
      author='Alex Mitrevski',
      author_email='aleksandar.mitrevski@h-brs.de',
      keywords='robotics robot_action_execution robot_explainability',
      packages=find_packages(exclude=['contrib', 'docs', 'tests']),
      project_urls={
          'Source': 'https://github.com/alex-mitrevski/explainable-robot-execution-models'
      })
