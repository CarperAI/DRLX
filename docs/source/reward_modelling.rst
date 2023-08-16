.. _reward_modelling:

Reward Modelling
================

Reward models are used to generate a reward signal to be used during RL training for an image generation model. Typically, they take an image and return some reward. Some may use prompts while generating reward, but this is not neccesary.
The library includes some toy rewards intended primarily for debugging.

Toy Rewards
-----------

.. automodule:: drlx.reward_modelling.toy_rewards
   :members:
   :undoc-members:
   :show-inheritance:

Aesthetics
----------

.. autoclass:: drlx.reward_modelling.aesthetics.Aesthetics
   :members:
   :undoc-members:
   :show-inheritance:

Pickscore (WIP)
----------------

.. automodule:: drlx.reward_modelling.pickscore
   :members:
   :undoc-members:
   :show-inheritance: