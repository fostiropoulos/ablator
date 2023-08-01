=============
API Reference
=============

This is the class and function reference of ablator. Please refer to
the tutorials for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.


:mod:`ablator.analysis`: Analysis module
========================================

Plot module
---------------------

.. currentmodule:: ablator
.. autosummary::
   analysis.plot.main.PlotAnalysis
   analysis.plot.cat_plot.Categorical
   analysis.plot.cat_plot.ViolinPlot
   analysis.plot.num_plot.Numerical
   analysis.plot.num_plot.LinearPlot
   analysis.plot.utils.parse_name_remap


Analysis Main classes
---------------------

.. currentmodule:: ablator
.. autosummary::

   analysis.main.Analysis

Analysis Results classes
------------------------

.. currentmodule:: ablator
.. autosummary::
   analysis.results.Results
   analysis.results.read_result





:mod:`ablator.config`: Config module
======================================

Base Config classes
---------------------

.. currentmodule:: ablator

.. autosummary::

   config.main.ConfigBase
   config.main.configclass

Prototype Config classes
------------------------

.. currentmodule:: ablator

.. autosummary::

   config.proto.ModelConfig
   config.proto.RunConfig
   config.proto.TrainConfig

Config Type classes
---------------------

.. currentmodule:: ablator

.. autosummary::

   config.types.Derived

HPO Config classes
---------------------

.. currentmodule:: ablator

.. autosummary::

   config.hpo.FieldType
   config.hpo.SearchSpace

Parallel Config classes
------------------------

.. currentmodule:: ablator

.. autosummary::
   
   config.mp.Optim
   config.mp.ParallelConfig
   config.mp.SearchAlgo

Config Utils functions
----------------------

.. currentmodule:: ablator

.. autosummary::
   
   config.utils.dict_hash
   config.utils.flatten_nested_dict







:mod:`ablator.main`: Main module
========================================

HPO Sampler module
---------------------

.. currentmodule:: ablator

.. autosummary::

   main.hpo.base.BaseSampler
   main.hpo.grid.GridSampler
   main.hpo.optuna.OptunaSampler

Main Model module
-----------------------

.. currentmodule:: ablator
.. autosummary::
   main.model.main.ModelBase
   main.model.wrapper.ModelWrapper

Experiment State module
-----------------------

.. currentmodule:: ablator
.. autosummary::
   main.state.state.ExperimentState
   main.state.store.TrialState

Prototype Trainer classes
--------------------------

.. currentmodule:: ablator
.. autosummary::
   main.proto.ProtoTrainer

Multi-process Trainer classes
-----------------------------

.. currentmodule:: ablator
.. autosummary::
   main.mp.ParallelTrainer
   main.mp.train_main_remote








:mod:`ablator.modules`: Extra modules
========================================

Metrics module
---------------------

.. currentmodule:: ablator

.. autosummary::

   modules.metrics.main.Metrics
   modules.metrics.stores.ArrayStore
   modules.metrics.stores.MovingAverage
   modules.metrics.stores.PredictionStore

Storage module
-----------------------

.. currentmodule:: ablator
.. autosummary::
   modules.storage.remote.RemoteConfig
   modules.storage.remote.run_cmd_wait

Optimizer classes
-----------------------

.. currentmodule:: ablator
.. autosummary::
   modules.optimizer.OptimizerArgs
   modules.optimizer.OptimizerConfig
   modules.optimizer.SGDConfig
   modules.optimizer.AdamWConfig
   modules.optimizer.AdamConfig
   modules.optimizer.get_parameter_names
   modules.optimizer.get_optim_parameters

Scheduler classes
-----------------------

.. currentmodule:: ablator
.. autosummary::

   modules.scheduler.SchedulerArgs
   modules.scheduler.SchedulerConfig
   modules.scheduler.StepLRConfig
   modules.scheduler.OneCycleConfig
   modules.scheduler.PlateuaConfig







:mod:`ablator.utils`: Utilities
========================================

Base utilities
-----------------------

.. currentmodule:: ablator
.. autosummary::
   utils.base.apply_lambda_to_iter
   utils.base.debugger_is_active
   utils.base.get_gpu_mem
   utils.base.get_latest_chkpts
   utils.base.get_lr
   utils.base.init_weights
   utils.base.is_oom_exception
   utils.base.iter_to_device
   utils.base.iter_to_numpy
   utils.base.num_format
   utils.base.parse_device
   utils.base.set_seed

File utilities
-----------------------

.. currentmodule:: ablator
.. autosummary::
   utils.file.clean_checkpoints
   utils.file.default_val_parser
   utils.file.dict_to_json
   utils.file.json_to_dict
   utils.file.make_sub_dirs
   utils.file.nested_set
   utils.file.save_checkpoint
   
