.. _trainers:

Trainers
========

DRLX provides a base trainer class and specific trainers for different methods. The base trainer class provides the basic functionalities such as setting up the optimizer, scheduler, and model, saving and loading checkpoints. The specific trainers extend the base trainer and implement the training process for the specific method.

BaseTrainer
------------

.. automodule:: drlx.trainer
   :members: BaseTrainer
   :undoc-members:
   :show-inheritance:

DDPOTrainer
-------------

.. automodule:: drlx.trainer.ddpo_trainer
   :members: DDPOTrainer
   :undoc-members:
   :show-inheritance:
