import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as tmp
from trainer.base import BaseTrainer
from trainer.config.run import DDPConfig
from trainer.modules.model.wrapper import ModelWrapper


class DDPTrainer(BaseTrainer):
    # NOTE: because of differences in effective batch_size when training in DDP.
    # it converges differently than training vanilla.
    def __init__(
        self,
        model,
        run_config: DDPConfig,
        description="",
    ):

        # NOTE: this is because we have to create the model in each process and we can not copy.
        super().__init__(model=model, run_config=run_config, description=description)
        self.run_config: DDPConfig
        if self.run_config.world_size is None:
            self.world_size = torch.cuda.device_count()
        else:
            self.world_size = self.run_config.world_size

        # NOTE: nccl does not currently work when two processes are using the same device
        if self.run_config.backend is None:
            if self.world_size <= torch.cuda.device_count():
                backend = "nccl"
            else:
                backend = "gloo"
        else:
            backend = self.run_config.backend.value

        self.backend = backend
        if self.backend == "nccl":
            assert self.world_size <= torch.cuda.device_count(), (
                "nccl backend can not run multiple processes in the same GPU."
                "Reduce world_size or change backend to gloo."
            )
        self.ip = self.run_config.ip
        self.port = self.run_config.port

        os.environ["MASTER_ADDR"] = self.ip
        os.environ["MASTER_PORT"] = self.port

    def _launch(self, rank):
        model: ModelWrapper = copy.deepcopy(self.model)
        dist.init_process_group(self.backend, rank=rank, world_size=self.world_size)
        num_gpus = torch.cuda.device_count()
        device = rank % num_gpus
        run_config: DDPConfig = copy.deepcopy(self.run_config)
        run_config.rank = rank
        run_config.train_config.device = device
        torch.cuda.synchronize()
        dist.barrier()
        model.train(run_config=run_config)
        dist.barrier()
        dist.destroy_process_group()
        model.sync()

    def launch(self):
        self.init_state()
        tmp.spawn(
            self._launch,
            nprocs=self.world_size,
            join=True,
        )
