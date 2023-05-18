from pathlib import Path
from alblator.modules.metrics.main import TrainMetrics
import numpy as np
import unittest

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
                moving_aux_metrics={"mean"},
            ),
            "Duplicate metric names with built-ins {'my_tag_mean'}",
        ),
        (
            lambda: TrainMetrics(
                batch_limit=30,
                memory_limit=None,
                evaluation_functions={"some": lambda x: np.mean(x)},
                moving_average_limit=moving_average_limit,
                tags=["my_tag"],
                static_aux_metrics={"my_tag_some": float("inf")},
                moving_aux_metrics={"mean"},
            ),
            "Duplicate metric names with built-ins {'my_tag_some'}",
        ),
        (
            lambda: TrainMetrics(
                batch_limit=30,
                memory_limit=None,
                evaluation_functions={"some": lambda x: np.mean(x)},
                moving_average_limit=moving_average_limit,
                tags=["my_tag"],
                static_aux_metrics={"my_tag_mean": float("inf")},
                moving_aux_metrics={"mean"},
            ),
            "Duplicate metric names with built-ins {'my_tag_mean'}",
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
    assert m.to_dict() == {
        "my_tag_mean": np.nan,
        "my_tag_ma_some": np.nan,
        "some": float("inf"),
    }
    assert_error_msg(
        lambda: m.update_ma_metrics({"ma_some": 0.1, "ma_some_2": 2}, tag="my_tag"),
        "There are difference in the class metrics: ['my_tag_ma_some'] and parsed metrics ['my_tag_ma_some', 'my_tag_ma_some_2']",
    )
    assert_error_msg(
        lambda: m.update_ma_metrics({"a": 0.1}, tag="my_tag"),
        "There are difference in the class metrics: ['my_tag_ma_some'] and parsed metrics ['my_tag_a']",
    )
    assert_error_msg(
        lambda: m.update_static_metrics({"some_2": 1}),
        "There are difference in the class metrics: ['some'] and updated metrics ['some_2']",
    )
    assert_error_msg(
        lambda: m.update_ma_metrics({"ma_some": ""}, tag="my_tag"),
        "Invalid MovingAverage value type <class 'str'>",
    )
    m.update_static_metrics({"some": ""})
    assert m.to_dict() == {"my_tag_ma_some": np.nan, "my_tag_mean": np.nan, "some": ""}

    m.update_ma_metrics({"ma_some": np.array([0])}, tag="my_tag")
    assert m.to_dict() == {"my_tag_ma_some": 0.0, "my_tag_mean": np.nan, "some": ""}

    for i in np.arange(moving_average_limit + 10):
        m.update_ma_metrics({"ma_some": int(i)}, tag="my_tag")
    assert m.to_dict() == {
        "my_tag_ma_some": np.mean(np.arange(10, moving_average_limit + 10)),
        "my_tag_mean": np.nan,
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
        m.update_ma_metrics({"ma_some": int(i)}, tag="my_tag")

    assert sys.getsizeof(m._get_ma("my_tag_ma_some").arr) < memory_limit
    assert m.to_dict() == {"my_tag_ma_some": 997.5, "my_tag_mean": np.nan, "some": 0}

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



class TestUpdateMetrics(unittest.TestCase):

    def test_update_static_metrics(self):
        # Create a model.
        model = Model()

        # Set the static metrics of the model.
        model.static_metrics = {"accuracy": 0.9, "loss": 0.1}

        # Update the static metrics of the model.
        update_static_metrics(model, {"accuracy": 0.95, "loss": 0.05})

        # Check that the static metrics of the model have been updated.
        self.assertEqual(model.accuracy, 0.95)
        self.assertEqual(model.loss, 0.05)

    def test_update_ma_metrics(self):
        # Create a model.
        model = Model()

        # Set the moving average metrics of the model.
        model.moving_average_metrics = {"accuracy": [0.8, 0.9], "loss": [0.2, 0.1]}

        # Update the moving average metrics of the model.
        update_ma_metrics(model, {"accuracy": 0.95, "loss": 0.05})

        # Check that the moving average metrics of the model have been updated.
        self.assertEqual(model.moving_average_metrics["accuracy"], [0.8, 0.9, 0.95])
        self.assertEqual(model.moving_average_metrics["loss"], [0.2, 0.1, 0.05])



if __name__ == "__main__":

    def assert_error_msg(fn, error_msg):
        try:
            fn()
            assert False
        except Exception as excp:
            if not error_msg == str(excp):
                raise excp

    test_metrics(assert_error_msg)
    unittest.main()
