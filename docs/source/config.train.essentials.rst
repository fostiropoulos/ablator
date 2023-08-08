Configurations for Training Essentials
======================================

In the process of training a model, there are essential components that are required. These
include the model itself, the optimizer, the scheduler, and the training setting (batch
size, number of epochs, the optimizer to be used, etc.).

Main Model Configuration
------------------------
.. autoclass:: ablator.config.proto.ModelConfig
   :noindex:
   :members:
   :show-inheritance:
   :exclude-members: config_class


Optimizer Configurations
------------------------

.. automodule:: ablator.modules.optimizer
   :noindex:
   :members: OptimizerConfig
   :show-inheritance:
   :exclude-members: config_class, make_optimizer

Scheduler Configurations
------------------------

.. autoclass:: ablator.modules.scheduler.SchedulerConfig
   :noindex:
   :members:
   :show-inheritance:
   :exclude-members: config_class, make_scheduler

Training Configurations
-----------------------

.. autoclass:: ablator.config.proto.TrainConfig
   :noindex:
   :members:
   :show-inheritance:
   :exclude-members: config_class

