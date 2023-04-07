# Examples of usage training on locally on a Machine

The examples in [run.py](run.py) can be used to prototype, develop and train locally on a VM. The code can be interchangebly be used to create a docker image and be used in a GPU cluster or other distributed settings.


## Run



`python run.py --config base_config.yaml [--train_mode [dp, ddp, mp]]`

1. train-mode -> `unspecified` trains a single experiment in a single process with `BaseTrainer`
2. train-mode -> `dp` trains a single  `DataParallel` experiment in a single process using `BaseTrainer`
3. train-mode -> `ddp` trains a single experiment using multiple processes with `DDPTrainer`
4. train-mode -> `ddp` trains multiple experiments using multiple processes with `ParallelTrainer`


## Config

Uncomment and modify `remote_save_config` logger to experiment with remote logging. This is essential if running dockerized experiments, as the state of the experiment has to be saved outside the docker image for persistance (or the experiment results are lost at the end of training)


## Model

[resnet.py](resnet.py) includes a basic model implementation. The example can be extended to work with more advanced configurations and training settings.