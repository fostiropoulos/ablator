from abc import abstractmethod
from typing import Callable, Dict, Optional


from trainer.config.main import ConfigBase, Enum, configclass
from trainer.modules.logging._tensorboard import TensorboardLogger


class LoggerType(Enum):
    tensorboard = "tensorboard"


@configclass
class LoggerArgs(ConfigBase):
    @abstractmethod
    def make(self, resume, uid, summary_dir):
        pass

@configclass
class LoggerConfig(ConfigBase):
    name: LoggerType
    arguments: LoggerArgs

    def __init__(self, name, arguments: Optional[Dict] = None):
        self.name = LoggerType(name)

        argument_cls: Callable = LOGGER_MAP[self.name]
        if arguments is None:
            arguments = {}
        self.arguments = argument_cls(**arguments)


@configclass
class TensorboardArgs(LoggerArgs):
    def make(self, resume, uid, summary_dir):
        return TensorboardLogger(summary_dir=summary_dir)


LOGGER_MAP: Dict[LoggerType, Callable] = {
    LoggerType.tensorboard: TensorboardArgs,
}
