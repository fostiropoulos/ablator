from abc import ABC, abstractmethod

from trainer.config.main import ConfigBase, configclass


@configclass
class LoggerConfig(ConfigBase):
    @abstractmethod
    def make_logger(self):
        raise NotImplementedError


class LoggerBase(ABC):
    @abstractmethod
    def _add_image(self, k, v, itr, dataformats="CHW"):
        pass

    @abstractmethod
    def _add_table(self, k, v, itr):
        pass

    @abstractmethod
    def _add_text(self, k, v, itr):
        pass

    @abstractmethod
    def _add_scalars(self, k, v, itr):
        pass

    @abstractmethod
    def _add_scalar(self, k, v, itr):
        pass

    @abstractmethod
    def _write_config(self, config: ConfigBase):
        pass

    @abstractmethod
    def _sync(self):
        """
        Must work for a single trial
        """
        pass
