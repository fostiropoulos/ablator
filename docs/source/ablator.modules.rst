Modules package
=======================

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   ablator.modules.metrics
   ablator.modules.storage

Submodules
----------

Optimizers module
--------------------------------

.. autoclass:: ablator.modules.optimizer.OptimizerArgs
   :members:
   :show-inheritance:

.. autoclass:: ablator.modules.optimizer.OptimizerConfig
   :members:
   :show-inheritance:

.. autoclass:: ablator.modules.optimizer.SGDConfig
   :members:
   :show-inheritance:

.. autoclass:: ablator.modules.optimizer.AdamWConfig
   :members:
   :show-inheritance:

.. autoclass:: ablator.modules.optimizer.AdamConfig
   :members:
   :show-inheritance:

.. autofunction:: ablator.modules.optimizer.get_parameter_names

.. autofunction:: ablator.modules.optimizer.get_optim_parameters

Schedulers module
--------------------------------

.. autoclass:: ablator.modules.scheduler.SchedulerArgs
   :members:
   :show-inheritance:

.. autoclass:: ablator.modules.scheduler.SchedulerConfig
   :members:
   :show-inheritance:

.. autoclass:: ablator.modules.scheduler.StepLRConfig
   :members:
   :show-inheritance:

.. autoclass:: ablator.modules.scheduler.OneCycleConfig
   :members:
   :show-inheritance:

.. autoclass:: ablator.modules.scheduler.PlateuaConfig
   :members:
   :show-inheritance:
