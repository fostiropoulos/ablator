.. _parallel_config:

Configurations for parallel models experiments
==============================================

One of the main features of Ablator is the ability to train and optimize
multiple models for hyperparameter optimization in parallel. The main
components of this feature are ``SearchSpace`` and ``ParallelConfig``.

.. autoclass:: ablator.config.hpo.SearchSpace
   :noindex:
   :members:
   :show-inheritance:
   :exclude-members: config_class, make_dict, to_str, parsed_value_range, make_paths, contains

.. autoclass:: ablator.config.mp.ParallelConfig
   :noindex:
   :members:
   :show-inheritance:
   :exclude-members: config_class
