import pytest
import subprocess
from pathlib import Path
import os
import platform
import json


def test_rclone_sync(tmp_path: Path):
    system = platform.system().lower()
    assert system in ['windows', 'darwin', 'linux'], "OS not supported"

    local_path = tmp_path.joinpath("local_path")
    local_path.mkdir()
    dest_path = tmp_path.joinpath("destination")
    dest_path.mkdir()

    test_file_name = f'{local_path}/test_rclone_file.txt'
    test_file_content = 'This is a test file to test rclone sync command'
    with open(test_file_name, 'w') as f:
        f.write(test_file_content)

    rclone_path = os.path.join("..", "..", "ablator", "rclone")
    print(rclone_path)
    try:
        if system == 'windows':
            subprocess.run([rclone_path, 'sync', str(local_path), str(dest_path)], check=True, shell=True)
        else:
            subprocess.run([rclone_path, 'sync', str(local_path), str(dest_path)], check=True)
    except subprocess.CalledProcessError:
        pytest.fail("Rclone sync command failed")

    # Check that the file was synced correctly
    synced_file = f'{dest_path}/test_rclone_file.txt'
    assert os.path.isfile(synced_file)

    # Check the content of the synced file
    with open(synced_file, 'r') as f:
        synced_file_content = f.read()
    assert synced_file_content == test_file_content

    os.remove(test_file_name)
    os.remove(synced_file)


def create_rclone_gcs_config(remote_name, project_number, service_account_file):
    config = {
        remote_name: {
            'type': 'google cloud storage',
            'project_number': project_number,
            'service_account_file': service_account_file,
            'object_acl': 'private',
            'bucket_acl': 'private',
            'location': 'us',
            'storage_class': 'STANDARD'
        }
    }

    config_str = '\n'.join(f'[{name}]\n' + '\n'.join(f'{k} = {v}' for k, v in settings.items()) for name, settings in config.items())

    rclone_conf_path = os.path.join("..", "..", "ablator", "rclone.conf")
    with open(rclone_conf_path, 'w') as f:
        f.write(config_str)


def test_rclone_sync_to_gcs(tmp_path: Path):
    # Create a rclone config file
    remote_name = 'gcs'
    service_account_file = os.path.join(os.getcwd(), "gcs_service_account.json")  # type: ignore
    create_rclone_gcs_config(remote_name, "deepusc-390522", service_account_file)

    system = platform.system().lower()
    assert system in ['windows', 'darwin', 'linux'], "OS not supported"

    local_path = tmp_path.joinpath("local_path")
    local_path.mkdir()

    remote_path = 'gcs:ablator-bucket/ken_test'

    # Create a test file
    test_file_name = f'{local_path}/test_rclone_file.txt'
    test_file_content = 'This is a test file to test rclone sync command'
    with open(test_file_name, 'w') as f:
        f.write(test_file_content)

    # Sync the file to the GCS using rclone
    rclone_path = os.path.join("..", "..", "ablator", "rclone")
    rclone_conf_path = os.path.join("..", "..", "ablator", "rclone.conf")
    try:
        if system == 'windows':
            subprocess.run([rclone_path, '--config', rclone_conf_path, 'sync', str(local_path), remote_path], check=True, shell=True)
        else:
            subprocess.run([rclone_path, '--config', rclone_conf_path, 'sync', str(local_path), remote_path], check=True)
    except subprocess.CalledProcessError:
        pytest.fail("Rclone sync command failed")

    # Download the file from GCS to a new local directory
    download_path = tmp_path.joinpath("download_path")
    download_path.mkdir()
    try:
        if system == 'windows':
            subprocess.run([rclone_path, '--config', rclone_conf_path, 'copy',
                           f'{remote_path}/test_rclone_file.txt', str(download_path)], check=True, shell=True)
        else:
            subprocess.run([rclone_path, '--config', rclone_conf_path, 'copy',
                           f'{remote_path}/test_rclone_file.txt', str(download_path)], check=True)
    except subprocess.CalledProcessError:
        pytest.fail("Rclone copy command failed")

    # Check the content of the downloaded file
    downloaded_file = f'{download_path}/test_rclone_file.txt'
    print(downloaded_file)
    assert os.path.isfile(downloaded_file)
    with open(downloaded_file, 'r') as f:
        downloaded_file_content = f.read()
    assert downloaded_file_content == test_file_content

    # Clean up
    os.remove(test_file_name)
    os.remove(downloaded_file)


if __name__ == "__main__":
    import shutil
    tmp_path = Path("C:\\tmp\\rclone_test")
    # test rclone sync on local
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True)
    test_rclone_sync(tmp_path)

    # tes rclone sync to GCS
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True)
    test_rclone_sync_to_gcs(tmp_path)
    pass
