Configuration module
====================

In ablator, the configuration system serves as a skeleton for an experiment
definition. Ablator leverages this skeleton to instantiate experiments with
the corresponding configurations.

Configuration module is divided into several submodules, all of which you can
use to define your experiment. Ablator is also flexible enough to allow you to
write your own configurations and adapt it to your experiment's needs.

.. toctree::
   :maxdepth: 2

   ablator.config.base
   ablator.config.types

   config.train.essentials
   config.train.single.experiment
   config.train.parallel.experiment
