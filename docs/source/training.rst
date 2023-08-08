Training module
---------------

Other building blocks of ablator are the training module, which launch the experiment that has been
configured with the configuration module. This module is composed of two classes:

- The training interface (``ModelWrapper``) which defines boiler-plate code for training and evaluating models.

- The trainers (``ProtoTrainer`` and ``ParallelTrainer``) launch the training and HPO process of the experiment.
  These trainers will use the model wrapper and the running configuration to launch the experiment.


.. toctree::
   :maxdepth: 2

   training.interface
   training.prototrainer
   training.mptrainer