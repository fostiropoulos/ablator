Main package
====================

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   ablator.main.model

Submodules
----------

Model Configuration module
---------------------------

.. automodule:: ablator.main.configs
   :members:
   :show-inheritance:

Multi-process Trainer module
----------------------------

.. automodule:: ablator.main.mp
   :members:
   :show-inheritance:

Prototype Trainer module
-------------------------

.. automodule:: ablator.main.proto
   :members:
   :show-inheritance:

Experiment and Optuna state module
----------------------------------

.. autoclass:: ablator.main.state.Base
   :show-inheritance:
   :exclude-members: __init__

.. autoclass:: ablator.main.state.OptunaState
   :members:
   :show-inheritance:

.. autoclass:: ablator.main.state.Trial
   :members:
   :show-inheritance:
   :exclude-members: __init__

.. autoclass:: ablator.main.state.TrialState
   :members:
   :show-inheritance:

.. autofunction:: ablator.main.state.augment_trial_kwargs

.. autofunction:: ablator.main.state.parse_metrics

.. autofunction:: ablator.main.state.sample_trial_params

Module contents
---------------

.. automodule:: ablator.main
   :members:
   :show-inheritance:
