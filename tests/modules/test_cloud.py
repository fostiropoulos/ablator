from pathlib import Path
from ablator.modules.storage.cloud import GcpConfig
from unittest import mock
import socket
import torch
from unittest.mock import patch
from ablator.modules.loggers.file import FileLogger
def assert_error_msg(fn, error_msg):
    try:
        fn()
        assert False
    except Exception as excp:
        if not error_msg == str(excp):
            raise excp


def write_rand_tensors(tmp_path: Path, n=2):
    tensors = []
    for i in range(n):
        a = torch.rand(100)
        torch.save(a, tmp_path.joinpath(f"t_{i}.pt"))
        tensors.append(a)
    return tensors


def load_rand_tensors(tmp_path: Path, n=2):
    tensors = []
    for i in range(n):
        a = torch.load(tmp_path.joinpath(f"t_{i}.pt"))
        tensors.append(a)
    return tensors


def assert_tensor_list_eq(a, b):
    assert all([all(_a == _b) for _a, _b in zip(a, b)])


def assert_tensor_list_diff(a, b):
    assert all([all(_a != _b) for _a, _b in zip(a, b)])


def mock_list_bucket_error_cmd(self, destination: str | None = None):
    destination = (
            Path(self.bucket) / destination
            if destination is not None
            else Path(self.bucket)
        )
    cmd = ["gsutil", "ls", f"gs://{destination}"]
    raise Exception(f"There was an error running `{' '.join(cmd)}`. "
            "Make sure gsutil is installed and that the destination exists. `CommandException: One or more URLs matched no objects.`")
def mock_list_bucket_empty(self, destination: str | None = None):
    return []


def mock_rsync_up(
        self,
        local_path: Path,
        remote_path: str,
        logger: FileLogger | None = None,
    ):
    pass

def test_gcp(tmp_path: Path, bucket: str = "gs://iordanis/"):
    rand_folder = f"{torch.rand(1).item()}"
    rand_destination = bucket + rand_folder
    # GcpConfig(bucket=rand_destination)._make_process(["gsutil", "-m", "rm", "-rf", rand_destination], verbose=False)


    with mock.patch("ablator.modules.storage.cloud.GcpConfig.list_bucket",mock_list_bucket_error_cmd):
        assert_error_msg(
            lambda: GcpConfig(bucket=rand_destination),
            f"There was an error running `gsutil ls {rand_destination}`. Make sure gsutil is installed and that the destination exists. `CommandException: One or more URLs matched no objects.`",
        )
    with mock.patch("ablator.modules.storage.cloud.GcpConfig._find_gcp_nodes",return_value={}):
        with mock.patch("ablator.modules.storage.cloud.GcpConfig.list_bucket",mock_list_bucket_empty):
            with mock.patch("socket.gethostname", return_value="localhost"):
                assert_error_msg(
                    lambda: GcpConfig(bucket=bucket),
                    f"Can only use GcpConfig from Google Cloud Server. Consider switching to RemoteConfig.",
                )
    
    
    with mock.patch("socket.gethostname", return_value="gcp-machine1"):
        with mock.patch("socket.gethostbyname",return_value="111.111.111.111"):
            with mock.patch("ablator.modules.storage.cloud.GcpConfig._find_gcp_nodes",return_value=[{"networkInterfaces":[{"networkIP":"111.111.111.111"}]}]):
                with mock.patch("ablator.modules.storage.cloud.GcpConfig.list_bucket",mock_list_bucket_empty):
                    with mock.patch("ablator.modules.storage.cloud.GcpConfig.rsync_up",mock_rsync_up):
                        cfg = GcpConfig(bucket=bucket)
                        files = cfg.list_bucket()
                        original_tensors = write_rand_tensors(tmp_path)
                        cfg.rsync_up(tmp_path, rand_folder)

    
    with mock.patch("ablator.modules.storage.cloud.GcpConfig.list_bucket",return_value={f"gs://{Path(cfg.bucket) / rand_folder}/"}):
        new_files = cfg.list_bucket()
    rand_destination = f"gs://{Path(cfg.bucket) / rand_folder}/"
    assert set(new_files).difference(files) == {rand_destination}
    uploaded_files=None
    with mock.patch("ablator.modules.storage.cloud.GcpConfig.list_bucket",return_value={f"{rand_destination}/tenosr1",f"{rand_destination}/tenosr2"}):
        uploaded_files = cfg.list_bucket(rand_folder + "/" + tmp_path.name)
    assert len(uploaded_files) == 2
    # Replace original tensors
    new_tensors = write_rand_tensors(tmp_path)
    assert_tensor_list_diff(original_tensors, new_tensors)
    loaded_tensors = load_rand_tensors(tmp_path=tmp_path)
    assert_tensor_list_eq(loaded_tensors, new_tensors)
    # Update the local tensors with the original tensors from gcp
    
    def mock_sync_down_new_file(self,remote_path: str,local_path: Path,logger: FileLogger | None = None,verbose=True):
        for i in range(2):
            torch.save(original_tensors[i],tmp_path.joinpath(f"t_{i}.pt"))
    with mock.patch("ablator.modules.storage.cloud.GcpConfig.rsync_down",mock_sync_down_new_file):
        cfg.rsync_down(rand_folder, tmp_path, verbose=False)
    loaded_tensors = load_rand_tensors(tmp_path=tmp_path)
    assert_tensor_list_eq(loaded_tensors, original_tensors)

    # Update a mock node from gcp
    with mock.patch("socket.gethostname", return_value="localhost"):
        hostname = socket.gethostname()
    mock_node_path = tmp_path.joinpath(hostname).joinpath(tmp_path.name)
    def mock_sync_down_node_new_file(self,node_hostname,remote_path: str,local_path: Path,logger: FileLogger | None = None,verbose=True):
        mock_node_path.parent.mkdir(exist_ok=True)
        mock_node_path.mkdir(exist_ok=True)
        for i in range(2):
            torch.save(original_tensors[i],mock_node_path.joinpath(f"t_{i}.pt"))
    with mock.patch("ablator.modules.storage.cloud.GcpConfig.rsync_down_node",mock_sync_down_node_new_file):
        cfg.rsync_down_node(hostname, rand_folder, mock_node_path)
    node_tensors = load_rand_tensors(tmp_path=mock_node_path)
    assert_tensor_list_eq(node_tensors, original_tensors)
    # TODO teardown refactoring
    cmd = ["gsutil", "-m", "rm", "-rf", rand_destination]
    
    with patch("subprocess.Popen") as mock_popen:
        mock_instance = mock_popen.return_value
        mock_instance.communicate.return_value = ("","error")
        p = cfg._make_process(cmd, verbose=False)
        out, err = p.communicate()
        assert len(out) == 0 and len(err) > 0


if __name__ == "__main__":
    import shutil

    bucket = "gs://iordanis/"

    rand_folder = f"aabb"
    rand_destination = bucket + rand_folder
    try:
        p = GcpConfig(bucket=rand_destination)._make_process(
            ["gsutil", "-m", "rm", "-rf", rand_destination], verbose=False
        )
        p.wait()
    except:
        pass
    tmp_path = Path("/tmp/gcp_test")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True)
    test_gcp(tmp_path, bucket)
    breakpoint()

    pass
