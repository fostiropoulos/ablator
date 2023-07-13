from pathlib import Path
from ablator.main.configs import GcsRcloneConfig
from ablator.modules.storage.rclone import RcloneConfig


def test_rclone_gcs(tmp_path: Path):
    gcs_config = GcsRcloneConfig(
        bucket="ablator_example_bucket_3",
        service_account_file="./hjzcerti.json",
    )
    gcs_config.startMount(tmp_path, verbose=False)


if __name__ == "__main__":
    test_rclone_gcs(Path("/mounttest"))
