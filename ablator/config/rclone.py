from ablator.config.main import configclass
from ablator.config.types import Stateless
from ablator.modules.storage.rclone import RcloneConfig


@configclass
class GcsRcloneConfig(RcloneConfig):
    config_name = "gcs"
    type: str = "google cloud storage"
    project_number: int = 0
    service_account_file: str
    bucket: str
    object_acl: str = "private"
    bucket_acl: str = "private"
    location: str = "us"
    storage_class: str = "STANDARD"

    def get_remote_path_prefix(self) -> str:
        return f"{self.config_name}:{self.bucket}/{self.remote_path}"


@configclass
class RemoteRcloneConfig(RcloneConfig):
    type: str = "sftp"
    config_name = "remote"
    host: Stateless[str]
    user: Stateless[str]
    port: Stateless[int] = 22
    key_file: Stateless[str]
