# Examples of usage training on locally on a Machine

The examples in [run.py](run.py) can be used to prototype, develop and train locally on a VM.
## TL;DR
To execute the example of a ResNet experiment you can run:

1. Locally: `python run.py --config config.yaml`

2. Distributed: `python run.py --config mp_config.yaml --mp`
## Run


To run you will need to supply the configuration and optionally the `--mp` flag. **NOTE** that you must be connected to a ray cluster to be able to run distributed. For getting started with Ray please read [here](https://docs.ray.io/en/latest/cluster/getting-started.html)

In summary and for simple use-cases you can mock a ray cluster by running on your local machine:

`ray start --head`

To execute the example of a ResNet experiment you can run:

1. Locally: `python run.py --config config.yaml`

2. Distributed: `python run.py --config mp_config.yaml --mp`


## Extending ResNet

[resnet.py](resnet.py) includes a basic model implementation. The example can be extended to work with more advanced configurations and training settings. Advanced use-case examples are in progress. Resnet.py provides an ablation experiment on different resnet models for learning rate and batch size.

#### Summary

As a **minimum** requirement to adapt the example to your use-case, you would need to:

### 1. Define your Configuration

```python
from ablator import ModelConfig, configclass

@configclass
class MyConfig(ModelConfig):
    # Configurable attributes
    ...
```

Create a yaml file [config.yaml](config.yaml):

```yaml
experiment_dir: "/tmp/dir"
train_config:
  dataset: cifar10
  optimizer_config:
    name: adam
    arguments:
      lr: 0.01
      weight_decay: 0.0
  batch_size: 128
  epochs: 10
  scheduler_config: null
  rand_weights_init: true
model_config:
    ...
```

### 2. Define your Model Class

```python
class MyModel(nn.Module):
    def __init__(self, config: MyConfig) -> None:
        ...
```

### 3. Overwrite the Dataloader


```python

from ablator import ModelWrapper
class MyModelWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: MyConfig):
        return load_cifar10(run_config, flag="train")
```

### 4. Run Locally

```python
wrapper = MyModelWrapper(
    model_class=MyModel,
)
run_config = ResRunConfig.load(config)
ablator = ProtoTrainer(
    wrapper=wrapper,
    run_config=run_config,
)
ablator.launch()
```

### 5. Run Distributed

Append to `config.yaml` ([example](mp_config.yaml))

```yaml
metrics_n_batches: 400
total_trials: 10
search_space:
  train_config.optimizer_config.arguments.lr:
    value_range: [0.0001,0.01]
    value_type: float
   ...
concurrent_trials: 10
optim_metrics:
  "val_loss": "min"
gpu_mb_per_experiment: 100
cpus_per_experiment: 1
```
Run:
```python
wrapper = MyModelWrapper(
    model_class=MyModel,
)
run_config = MyParallelConfig.load(mp_config)
ablator = ParallelTrainer(
    wrapper=wrapper,
    run_config=run_config,
)

ablator.launch(
    working_directory=WORKING_DIRECTORY
)
```