import copy
import io
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest
import json
from PIL import Image

from ablator import ModelConfig, OptimizerConfig, RunConfig, TrainConfig
from ablator.modules.loggers.main import SummaryLogger


def assert_console_output(fn, assert_fn):
    f = io.StringIO()
    with redirect_stdout(f):
        fn()
    s = f.getvalue()
    assert assert_fn(s)


def assert_error_msg_fn(fn, error_msg_fn):
    try:
        fn()
        assert False
    except Exception as excp:
        if not error_msg_fn(str(excp)):
            print(str(excp))
            raise excp


def assert_error_msg(fn, error_msg):
    try:
        fn()
        assert False
    except Exception as excp:
        if not error_msg == str(excp):
            print(str(excp))
            raise excp


def assert_iter_equals(l1, l2):
    assert sorted(list(l1)) == sorted(list(l2))


model_c = ModelConfig()
train_c = TrainConfig(
    dataset="x",
    batch_size=128,
    epochs=1,
    optimizer_config=OptimizerConfig(name="sgd", arguments={"lr": 0.1}),
)

c = RunConfig(model_config=model_c, train_config=train_c)


def test_summary_logger(tmp_path: Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # logpath = tmp_path.joinpath("test.log")
    # logpath.unlink()
    tmp_path = tmp_path.joinpath(f"{random.random()}")
    l = SummaryLogger(c, tmp_path)
    assert_error_msg(
        lambda: SummaryLogger(c, tmp_path),
        f"SummaryLogger: Resume is set to False but {tmp_path} exists.",
    )
    assert l.uid == c.uid
    assert_iter_equals(
        [p.name for p in tmp_path.glob("*")],
        SummaryLogger.CHKPT_DIR_VALUES
        + [
            SummaryLogger.CONFIG_FILE_NAME,
            SummaryLogger.SUMMARY_DIR_NAME,
            SummaryLogger.LOG_FILE_NAME,
            SummaryLogger.METADATA_JSON,
        ],
    )
    c2 = copy.deepcopy(c)
    c2.train_config.dataset = "b"
    assert_error_msg_fn(
        lambda: SummaryLogger(c2, tmp_path, resume=True),
        lambda msg: msg.startswith(
            "Differences between configurations:"
        ),
    )
    l = SummaryLogger(c, tmp_path, resume=True)
    save_dict = {"A": np.random.random(100)}
    l.checkpoint(save_dict, "b")

    l = SummaryLogger(c, tmp_path, resume=True)
    l.checkpoint(save_dict, "b")

    assert_error_msg(
        lambda: l.checkpoint(save_dict, "b", itr=0),
        f"Checkpoint iteration 1 >= training iteration 0. Can not overwrite checkpoint.",
    )
    del l.checkpoint_iteration["recent"]["b"]
    assert_error_msg(
        lambda: l.checkpoint(save_dict, "b", itr=0),
        f"Checkpoint exists: {tmp_path.joinpath('checkpoints/b_0000000000.pt')}",
    )
    assert_error_msg(
        lambda: l.checkpoint(save_dict, "b", itr=1),
        f"Checkpoint exists: {tmp_path.joinpath('checkpoints/b_0000000001.pt')}",
    )
    assert len(list(tmp_path.joinpath("checkpoints").glob("*.pt"))) == 2

    l.checkpoint(save_dict, "b", 10)
    l.checkpoint(save_dict, "b", 12)
    l.checkpoint(save_dict, "b", 15)
    l = SummaryLogger(c, tmp_path, resume=True, keep_n_checkpoints=1)
    assert len(list(tmp_path.joinpath("checkpoints").glob("*.pt"))) == 5
    l.checkpoint(save_dict, "b", 16)
    assert len(list(tmp_path.joinpath("checkpoints").glob("*16.pt"))) == 1
    l = SummaryLogger(c, tmp_path, resume=True, keep_n_checkpoints=3)
    for i in range(100):
        l.checkpoint(save_dict, "b", is_best=True)

    assert len(list(tmp_path.joinpath("best_checkpoints").glob("*.pt"))) == 3
    assert sorted(list(tmp_path.joinpath("best_checkpoints").glob("*.pt")))[
        -1
    ] == tmp_path.joinpath("best_checkpoints", "b_0000000099.pt")
    l.clean_checkpoints(0)
    assert len(list(tmp_path.joinpath("best_checkpoints").glob("*.pt"))) == 0

    l = SummaryLogger(c, tmp_path, resume=True, keep_n_checkpoints=3)

    def wait_for_tensorboard(event_acc, tag, tag_type="scalars", max_wait_time=50, output_fn=False):
        start_time = time.time()
        while True:
            l.dashboard.backend_logger.flush()
            if time.time() - start_time > max_wait_time:
                raise RuntimeError(f"Timed out waiting for {tag} to appear in TensorBoard.")
            event_acc.Reload()
            if output_fn:
                print(event_acc.Tags())
            if tag in event_acc.Tags()[tag_type]:
                break
            time.sleep(0.1)

    def wait_for_tensorboard_update(event_acc, tag, expected_value, tag_type="scalars", max_wait_time=50, output_fn=False):
        start_time = time.time()
        while True:
            l.dashboard.backend_logger.flush()
            if time.time() - start_time > max_wait_time:
                raise RuntimeError(f"Timed out waiting for the latest value of {tag} to appear in TensorBoard.")
            event_acc.Reload()
            if tag in event_acc.Tags()[tag_type]:
                event_list = event_acc.Scalars(tag)
                if output_fn:
                    print(event_list[-1])
                if event_list[-1].value == expected_value:
                    return True
            time.sleep(0.1)

    event_acc = EventAccumulator(
        tmp_path.joinpath("dashboard", "tensorboard").as_posix()
    )
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]
    assert len(tags) == 0
    l.update({"test": 0})
    l.dashboard.backend_logger.flush()
    wait_for_tensorboard(event_acc, "test")
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]
    assert len(tags) == 1

    event_list = event_acc.Scalars("test")
    assert len(event_list) == 1
    assert event_list[0].step == 0
    assert event_list[0].value == 0

    l.update({"test": 5})
    l.dashboard.backend_logger.flush()
    wait_for_tensorboard_update(event_acc, "test", 5)
    l.update({"test": 10}, itr=100)
    l.dashboard.backend_logger.flush()
    wait_for_tensorboard_update(event_acc, "test", 10)
    event_acc.Reload()
    event_list = event_acc.Scalars("test")
    assert event_list[1].step == 1
    assert event_list[1].value == 5
    assert event_list[2].step == 100
    assert event_list[2].value == 10

    l.update({"test_arr": np.array([100, "100"])})
    l.dashboard.backend_logger.flush()
    wait_for_tensorboard(event_acc, "test_arr/text_summary", tag_type="tensors")
    event_acc.Reload()
    assert str(event_acc.Tensors("test_arr/text_summary")[0]).endswith(
        'dtype: DT_STRING\ntensor_shape {\n  dim {\n    size: 1\n  }\n}\nstring_val: "100 100"\n)'
    )

    l.update({"arr": np.array([100, 101])})
    l.dashboard.backend_logger.flush()
    wait_for_tensorboard(event_acc, "arr_0", tag_type="scalars")
    wait_for_tensorboard(event_acc, "arr_1", tag_type="scalars")
    event_acc.Reload()

    event_list = event_acc.Scalars("arr_0")
    assert event_list[0].step == 101
    assert event_list[0].value == 100
    event_list = event_acc.Scalars("arr_1")
    assert event_list[0].step == 101
    assert event_list[0].value == 101

    l.update({"text": "bb"})
    l.dashboard.backend_logger.flush()
    wait_for_tensorboard(event_acc, "text/text_summary", tag_type="tensors")

    event_acc.Reload()
    assert str(event_acc.Tensors("text/text_summary")[0]).endswith(
        'string_val: "bb"\n)'
    )

    l.update({"df": pd.DataFrame(np.zeros(3))})
    l.dashboard.backend_logger.flush()
    wait_for_tensorboard(event_acc, "df/text_summary", tag_type="tensors")
    event_acc.Reload()
    assert str(event_acc.Tensors("df/text_summary")[0]).endswith(
        'string_val: "|    |   0 |\\n|---:|----:|\\n|  0 |   0 |\\n|  1 |   0 |\\n|  2 |   0 |"\n)'
    )

    img = Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8))
    l.update({"img": img})
    l.dashboard.backend_logger.flush()
    wait_for_tensorboard(event_acc, "img", tag_type="images")
    event_acc.Reload()

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    assert event_acc.Images("img")[0].encoded_image_string == img_byte_arr


def test_results_json(tmp_path: Path):
    tmp_path = tmp_path.joinpath(f"{random.random()}")
    l = SummaryLogger(c, tmp_path)
    for i in range(10):
        df = pd.DataFrame(np.random.rand(3))
        l.update({"df": df})
        results = json.loads(l.result_json_path.read_text())
        assert (df == pd.DataFrame(results[-1]["df"])).all().all()
    assert len(results) == 10

    l.update({"test": 5})
    results = json.loads(l.result_json_path.read_text())
    assert results[-1]["test"] == 5
    assert "df" not in results[-1]
    l.update({"test": "10"})
    results = json.loads(l.result_json_path.read_text())
    assert results[-1]["test"] == "10"
    # breakpoint()
    # return
    pass


if __name__ == "__main__":
    # test_results_json(Path("/tmp/"))
    test_summary_logger(Path("/tmp/"))

    pass
