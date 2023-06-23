import os
import sys
import platform
import urllib.request

import zipfile
import glob
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = os.path.join(CURRENT_DIR, "..",  "ablator/")
PATH_EXTRACT_ZIP_TO = os.path.join(CURRENT_DIR, "..",  "rclone_installing/")


def get_system_architecture():
    system = platform.system().lower()
    machine = platform.machine()

    if system == "windows":
        arch = "amd64" if machine in ["AMD64"] else "386"
    elif system == "darwin":
        system = "osx"
        arch = "amd64" if machine in ["x86_64", "amd64"] else "386"
    elif system == "linux":
        arch = "amd64" if machine in ["x86_64", "amd64"] else "386"
    else:
        logging.error("Unsupported OS type")
        sys.exit(2)

    return system, arch


def get_rclone_download_url(system, arch, beta=False):
    if beta:
        download_url = f"https://beta.rclone.org/rclone-beta-latest-{system}-{arch}.zip"
        rclone_zip = f"rclone-beta-latest-{system}-{arch}.zip"
    else:
        download_url = f"https://downloads.rclone.org/rclone-current-{system}-{arch}.zip"
        rclone_zip = f"rclone-current-{system}-{arch}.zip"

    return download_url, rclone_zip


def find_rclone(known_part, unknown_part):
    logging.info(f"Searching for rclone in {known_part}")
    pattern = os.path.join(known_part, unknown_part)
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def install_rclone_on_windows(path_extract_to, project_path):
    try:
        rclone_exe = find_rclone(path_extract_to, "*/rclone.exe")
        assert rclone_exe is not None, logging.error("rclone.exe not found in extracted zip file")
        shutil.move(rclone_exe, f"{project_path}//rclone.exe")
    except PermissionError:
        logging.error("Permission denied. You might need to run this script as Administrator.")


def install_rclone_on_Linux(path_extract_to, project_path):
    try:
        rclone = find_rclone(path_extract_to, "*/rclone")
        assert rclone is not None, logging.error("rclone not found in extracted zip file")
        shutil.move(rclone, f"{project_path}/rclone")
        os.chmod(f"{project_path}/rclone", 0o755)
    except PermissionError:
        logging.error("Permission denied. You might need to run this script as root.")


def create_rclone_config(rclone_env_path):
    dir_path = rclone_env_path
    config_filename = "rclone.conf"
    config = {
        "gcs": {
            'type': 'google cloud storage',
            'project_number': "project_number",
            'service_account_file': "service_account_file",
            'object_acl': 'private',
            'bucket_acl': 'private',
            'location': 'us',
            'storage_class': 'STANDARD'
        }
    }
    config_content = '\n'.join(f'[{name}]\n' + '\n'.join(f'{k} = {v}' for k, v in settings.items()) for name, settings in config.items())

    os.makedirs(dir_path, exist_ok=True)
    config_path = os.path.join(dir_path, config_filename)
    # Why I create a file to set config rather than using tempfiler.NamedTemporaryFile
    # Cause it has bug on windows, see https://bugs.python.org/issue14243
    with open(config_path, "w") as f:
        f.write(config_content)
    logging.info(f"Rclone configuration file has been written to {config_path}")


def download_and_install_rclone(beta=False):
    logging.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logging.info(f"PROJECT_PATH: {PROJECT_PATH}")
    logging.info(f"PATH_EXTRACT_ZIP_TO: {PATH_EXTRACT_ZIP_TO}")
    logging.info('download_and_install_rclone started')

    # Downloads and installs rclone based on the OS type and architecture
    system, arch = get_system_architecture()

    # Get download url and zip file name
    download_url, rclone_zip = get_rclone_download_url(system, arch, beta)

    # Download rclone zip file and save it to "rclone_zip" directory
    logging.info(f"download_url: {download_url}")
    print(f"Downloading rclone from {download_url}...")
    rclone_zip = os.path.join(CURRENT_DIR, "..",  rclone_zip)
    if not os.path.exists(rclone_zip):
        urllib.request.urlretrieve(download_url, rclone_zip)

    # Unzip rclone zip file
    logging.info(f"Unzipping {rclone_zip}...")
    with zipfile.ZipFile(rclone_zip, 'r') as zip_ref:
        zip_ref.extractall(PATH_EXTRACT_ZIP_TO)

    # Move rclone binary to appropriate directory
    if system == "windows":
        install_rclone_on_windows(PATH_EXTRACT_ZIP_TO, PROJECT_PATH)
    else:
        install_rclone_on_Linux(PATH_EXTRACT_ZIP_TO, PROJECT_PATH)

    # Create rclone default configuration file
    logging.info("Creating rclone configuration file...")
    create_rclone_config(PROJECT_PATH)

    # Clean up
    logging.info("Cleaning up...")
    shutil.rmtree(PATH_EXTRACT_ZIP_TO)
    os.remove(rclone_zip)


if __name__ == "__main__":
    download_and_install_rclone()
    logging.info('download_and_install_rclone Finsihed')
    exit(0)
