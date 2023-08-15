.. _denoisers:

Denoisers
=========

DRLX generally uses conditioned denoisers for diffusion modelling. Currently, the library is made with text conditioning in mind, the base classes are with generalizability in mind, and to this end the conditional denoiser
supports any kind of conditioning signal that produces an embedding.

BaseConditionalDenoiser
-------------------------

.. automodule:: drlx.denoisers
   :members:
   :undoc-members:
   :show-inheritance:

LDMUNet
----------

.. automodule:: drlx.denoisers.ldm_unet
   :members:
   :undoc-members:
   :show-inheritance:
