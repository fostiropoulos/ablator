from pathlib import Path
from trainer.modules.storage.cloud import GcpConfig
from unittest import mock
import socket
import torch


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


def test_gcp(tmp_path: Path, bucket: str = "gs://iordanis-data/"):
    rand_folder = f"{torch.rand(1).item()}"
    rand_destination = bucket + rand_folder
    # GcpConfig(bucket=rand_destination)._make_process(["gsutil", "-m", "rm", "-rf", rand_destination], verbose=False)
    assert_error_msg(
        lambda: GcpConfig(bucket=rand_destination),
        f"There was an error running `gsutil ls {rand_destination}`. Make sure gsutil is installed and that the destination exists. `CommandException: One or more URLs matched no objects.`",
    )
    with mock.patch("socket.gethostname", return_value="localhost"):
        assert_error_msg(
            lambda: GcpConfig(bucket=bucket),
            f"Can only use GcpConfig from Google Cloud Server. Consider switching to RemoteConfig.",
        )
    cfg = GcpConfig(bucket=bucket)
    files = cfg.list_bucket()
    original_tensors = write_rand_tensors(tmp_path)
    cfg.rsync_up(tmp_path, rand_folder)

    new_files = cfg.list_bucket()
    rand_destination = f"gs://{Path(cfg.bucket) / rand_folder}/"
    assert set(new_files).difference(files) == {rand_destination}
    uploaded_files = cfg.list_bucket(rand_folder + "/" + tmp_path.name)
    assert len(uploaded_files) == 2
    # Replace original tensors
    new_tensors = write_rand_tensors(tmp_path)
    assert_tensor_list_diff(original_tensors, new_tensors)
    loaded_tensors = load_rand_tensors(tmp_path=tmp_path)
    assert_tensor_list_eq(loaded_tensors, new_tensors)
    # Update the local tensors with the original tensors from gcp
    cfg.rsync_down(rand_folder, tmp_path, verbose=False)
    loaded_tensors = load_rand_tensors(tmp_path=tmp_path)
    assert_tensor_list_eq(loaded_tensors, original_tensors)

    # Update a mock node from gcp
    hostname = socket.gethostname()
    mock_node_path = tmp_path.joinpath(hostname).joinpath(tmp_path.name)
    cfg.rsync_down_nodes(hostname, rand_folder, mock_node_path)
    node_tensors = load_rand_tensors(tmp_path=mock_node_path)
    assert_tensor_list_eq(node_tensors, original_tensors)
    # TODO teardown refactoring
    cmd = ["gsutil", "-m", "rm", "-rf", rand_destination]
    p = cfg._make_process(cmd, verbose=False)
    out, err = p.communicate()
    assert len(out) == 0 and len(err) > 0


if __name__ == "__main__":
    import shutil

    bucket = "gs://iordanis-data/"

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
    shutil.rmtree(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    test_gcp(tmp_path, bucket)
    breakpoint()

    pass
