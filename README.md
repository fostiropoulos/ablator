# Model Trainer

A distributed experiment execution framework for ablation studies. Model Trainer provides a wrapper for your model and a Trainer class for you to prototype on your method and scale to thousands of experimental trials with 1 code change.

Ablation studies are experiments used to identify the causal effects on a method performance. Method is a `meta-model` from a Bayesian point of view, where the model which we evaluate include the model parameters as well as meta-training arguments, such as Optimizer. In plain english, the causal effects we study encompass both the PyTorch model and the training configuration.

## Why Model Trainer?
 1. Strictly typed configuration system prevents errors.
 2. Seamless prototyping to production
 3. Stateful experiment design. Stop, Resume, Share your experiments
 4. Automated analysis artifacts
 5. Template Training

### What is the difference with using `xxx`

In summary, you will need to integrate different tools, for distributed execution, fault tollerance, training, checkpointing and analysis. Poor compatibility between tools, verisioning errors will lead to errors in your analysis.


You can use Model Trainer with any other library i.e. PyTorch Lighting. Just wrap a Lighting model with ModelWrapper


Spend more time in the creative process of ML research and less time on dev-ops.

### Alpha - Phase

The library is under active development and a lot of the API endpoints will be removed / renamed or their functionality changed without notice.

## High-Level Overview

### 1. Install

Use a python virtual enviroment to avoid version conflicts.

1. `git clone git@github.com:fostiropoulos/trainer.git`
2. `cd trainer`
3. `pip install .`

For Development
`pip install .[dev]`
### 2. TODO