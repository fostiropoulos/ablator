import os
import sys
import platform
import requests
import zipfile
import glob
import shutil
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_NAME = "ablator"
PATH_EXTRACT_ZIP_TO = "./rclone_installing/"


def get_rclone_download_url(system, machine, beta=False):
    """Returns the download url for rclone based on the OS type and architecture
    """
    if system == "windows":
        arch = "amd64" if machine in ["AMD64"] else "386"
    elif system == "darwin":
        system = "osx"
        arch = "amd64" if machine in ["x86_64", "amd64"] else "386"
    elif system == "linux":
        arch = "amd64" if machine in ["x86_64", "amd64"] else "386"
    else:
        print("OS type not supported")
        sys.exit(2)

    if beta:
        download_url = f"https://beta.rclone.org/rclone-beta-latest-{system}-{arch}.zip"
        rclone_zip = f"rclone-beta-latest-{system}-{arch}.zip"
    else:
        download_url = f"https://downloads.rclone.org/rclone-current-{system}-{arch}.zip"
        rclone_zip = f"rclone-current-{system}-{arch}.zip"

    return download_url, rclone_zip


def find_rclone(known_part, unknown_part):
    logging.info(f"Searching for rclone in {known_part}")
    pattern = known_part + unknown_part
    matches = glob.glob(pattern)
    return matches[0]


def install_rclone_on_windows(path_extract_to):
    try:
        shutil.move(find_rclone(path_extract_to, "*/rclone.exe"), f"./{PROJECT_NAME}//rclone.exe")
    except PermissionError:
        logging.error("Permission denied. You might need to run this script as Administrator.")


def install_rclone_on_Linux(path_extract_to):
    try:
        shutil.move(find_rclone(path_extract_to, "*/rclone"), f"./{PROJECT_NAME}/rclone")
        os.chmod(f"./{PROJECT_NAME}/rclone", 0o755)
    except PermissionError:
        logging.error("Permission denied. You might need to run this script as root.")


def create_rclone_config(rclone_env_path):
    dir_path = rclone_env_path  # "./ablator/rclone"
    config_filename = "rclone.conf"
    config_content = """
    [minio]
    type = s3
    provider = Minio
    env_auth = false
    access_key_id = YOUR_ACCESS_KEY
    secret_access_key = YOUR_SECRET_KEY
    endpoint = https://YOUR_MINIO_ENDPOINT
    location_constraint =
    acl = private
    """
    # Ensure directory exists
    os.makedirs(dir_path, exist_ok=True)
    config_path = os.path.join(dir_path, config_filename)
    with open(config_path, "w") as f:
        f.write(config_content)
    logging.info(f"Rclone configuration file has been written to {config_path}")


def download_and_install_rclone(beta=False):
    """Downloads and installs rclone based on the OS type and architecture"""
    logging.info('download_and_install_rclone started')
    system = platform.system().lower()
    machine = platform.machine()

    # Get download url and zip file name
    download_url, rclone_zip = get_rclone_download_url(system, machine, beta)

    # Download rclone zip file and save it to "rclone_zip" directory
    logging.info(f"download_url: {download_url}")
    print(f"Downloading rclone from {download_url}...")
    if not os.path.exists(rclone_zip):
        resp = requests.get(download_url)
        with open(rclone_zip, 'wb') as f_out:
            f_out.write(resp.content)

    # Unzip rclone zip file
    logging.info(f"Unzipping {rclone_zip}...")
    path_extract_to = PATH_EXTRACT_ZIP_TO
    with zipfile.ZipFile(rclone_zip, 'r') as zip_ref:
        zip_ref.extractall(path_extract_to)

    # Move rclone binary to appropriate directory
    if system == "windows":
        install_rclone_on_windows(path_extract_to)
    else:
        install_rclone_on_Linux(path_extract_to)

    # Create rclone default configuration file
    logging.info("Creating rclone configuration file...")
    create_rclone_config(f"./{PROJECT_NAME}/")

    # Clean up
    logging.info("Cleaning up...")
    shutil.rmtree(path_extract_to)
    os.remove(rclone_zip)


if __name__ == "__main__":
    download_and_install_rclone()
    logging.info('download_and_install_rclone Finsihed')
    exit(0)
