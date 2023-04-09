from pathlib import Path
from trainer.modules.metrics.main import TrainMetrics
import numpy as np

import sys


moving_average_limit = 100
memory_limit = 100


def test_metrics(assert_error_msg):
    error_init = [
        (
            lambda: TrainMetrics(
                batch_limit=30,
                memory_limit=None,
                evaluation_functions={"mean": lambda x: np.mean(x)},
                moving_average_limit=moving_average_limit,
                tags=["my_tag"],
                static_aux_metrics={"some": float("inf")},
                moving_aux_metrics={"some"},
            ),
            "Overlapping metric names with built-ins {'some'}",
        ),
        (
            lambda: TrainMetrics(
                batch_limit=30,
                memory_limit=None,
                evaluation_functions={"mean": lambda x: np.mean(x)},
                moving_average_limit=moving_average_limit,
                tags=["my_tag"],
                static_aux_metrics={"some": float("inf")},
                moving_aux_metrics={"my_tag_mean"},
            ),
            "Overlapping metric names with built-ins {'my_tag_mean'}",
        ),
    ]
    for error_obj, error_msg in error_init:
        assert_error_msg(error_obj, error_msg)

    m = TrainMetrics(
        batch_limit=30,
        memory_limit=None,
        evaluation_functions={"mean": lambda x: np.mean(x)},
        moving_average_limit=moving_average_limit,
        tags=["my_tag"],
        static_aux_metrics={"some": float("inf")},
        moving_aux_metrics={"ma_some"},
    )
    assert m.to_dict() == {"my_tag_mean": None, "ma_some": None, "some": float("inf")}
    assert_error_msg(
        lambda: m.update_ma_metrics({"ma_some": 0.1, "ma_some_2": 2}),
        "There are difference in the class metrics: ['ma_some'] and updated metrics ['ma_some', 'ma_some_2']",
    )
    assert_error_msg(
        lambda: m.update_ma_metrics({"a": 0.1}),
        "There are difference in the class metrics: ['ma_some'] and updated metrics ['a']",
    )
    assert_error_msg(
        lambda: m.update_static_metrics({"some_2": 1}),
        "There are difference in the class metrics: ['some'] and updated metrics ['some_2']",
    )
    assert_error_msg(
        lambda: m.update_ma_metrics({"ma_some": ""}),
        "Invalid MovingAverage value type <class 'str'>",
    )
    m.update_static_metrics({"some": ""})
    assert m.to_dict() == {"ma_some": None, "my_tag_mean": None, "some": ""}

    m.update_ma_metrics({"ma_some": np.array([0])})
    assert m.to_dict() == {"ma_some": 0.0, "my_tag_mean": None, "some": ""}

    for i in np.arange(moving_average_limit + 10):
        m.update_ma_metrics({"ma_some": int(i)})
    assert m.to_dict() == {
        "ma_some": np.mean(np.arange(10, moving_average_limit + 10)),
        "my_tag_mean": None,
        "some": "",
    }

    m = TrainMetrics(
        batch_limit=30,
        memory_limit=memory_limit,
        evaluation_functions={"mean": lambda labels, preds: "a"},
        moving_average_limit=100,
        tags=["my_tag"],
        static_aux_metrics={"some": 0},
        moving_aux_metrics={"ma_some"},
    )
    for i in range(1000):
        m.update_ma_metrics({"ma_some": int(i)})

    assert sys.getsizeof(m._get_ma("ma_some").arr) < memory_limit
    assert m.to_dict() == {"ma_some": 997.5, "my_tag_mean": None, "some": 0}

    assert_error_msg(
        lambda: m.append_batch(1, preds="", labels=None, tag=""),
        "Metrics.append_batch takes no positional arguments.",
    )
    assert_error_msg(
        lambda: m.append_batch(preds="", labels="", tag=""),
        "Undefined tag ''. Metric tags ['my_tag']",
    )
    assert_error_msg(
        lambda: [
            m.append_batch(preds=np.array([""]), labels=np.array([""]), tag="my_tag"),
            m.evaluate("my_tag"),
        ],
        "Invalid value a returned by evaluation function <lambda>. Must be numeric scalar.",
    )
    assert_error_msg(
        lambda: [
            m.append_batch(preds=np.array([""]), labels=np.array([""]), tag="my_tag"),
            m.append_batch(preds=np.array([""]), tag="my_tag"),
        ],
        "Missing keys from the prediction store update. Expected: ['labels', 'preds'], received ['preds']",
    )
    assert_error_msg(
        lambda: [
            m.append_batch(preds=np.array([""]), labels=np.array([""]), tag="my_tag"),
            m.append_batch(
                preds=np.array([""]), labels=np.array([""] * 2), tag="my_tag"
            ),
        ],
        "Different number of batches between inputs. Sizes: {'preds': 1, 'labels': 2}",
    )
    m2 = TrainMetrics(
        batch_limit=30,
        memory_limit=memory_limit,
        evaluation_functions={"mean": lambda somex: np.mean(somex)},
        moving_average_limit=100,
        tags=["my_tag"],
        static_aux_metrics={"some": 0},
        moving_aux_metrics={"ma_some"},
    )
    assert_error_msg(
        lambda: [
            m2.append_batch(
                somex=np.array([100]), labels=np.array([1000]), tag="my_tag"
            ),
            m2.evaluate("my_tag"),
        ],
        "Evaluation function arguments ['somex'] different than stored predictions: ['labels', 'somex']",
    )
    m3 = TrainMetrics(
        batch_limit=30,
        memory_limit=None,
        evaluation_functions={"mean": lambda somex: np.mean(somex)},
        moving_average_limit=100,
        tags=["my_tag"],
    )
    assert_error_msg(
        lambda: [
            m3.evaluate("my_tag"),
        ],
        "PredictionStore has no predictions to evaluate.",
    )
    m3.append_batch(somex=np.array([100]), tag="my_tag")
    m3.evaluate("my_tag", reset=False, update_ma=True)
    m3.append_batch(somex=np.array([0] * 3), tag="my_tag")

    m3.evaluate("my_tag", reset=False, update_ma=False)
    assert m3.to_dict() == {"my_tag_mean": 100.0}
    m3.evaluate("my_tag", reset=False, update_ma=True)
    assert m3.to_dict() == {"my_tag_mean": 62.5}
    m3.append_batch(somex=np.array([0] * 3), tag="my_tag")
    assert m3.to_dict() == {"my_tag_mean": 62.5}
    m3.evaluate("my_tag", reset=False, update_ma=True)
    assert np.isclose(m3.to_dict()["my_tag_mean"], 46.42857142857142)


if __name__ == "__main__":
    from ..conftest import assert_error_msg
    test_metrics(assert_error_msg)
