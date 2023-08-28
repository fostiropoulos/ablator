Analysis module
---------------

In addition to the automatically generated TensorBoard results,
which shows training progress of all trials in the experiment,
ablator also provides a range of plotting tools that you can use
to facilitate the visualization of ablation experiment outcomes.

The analysis module has tools that allow you to observe the
correlation between the studied hyperparameters and the model's
performance. These visualization capabilities are done using
``ablator.analysis.plot.main.PlottingAnalysis`` and
``ablator.analysis.results.Results``.

.. toctree::
   :maxdepth: 2

   ablator.analysis.plot.main
   ablator.analysis.results