from ablator.modules.metrics.stores import ArrayStore, MovingAverage, PredictionStore
import sys
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def test_array_store(assert_error_msg):
    batch_limit = 50
    memory_limit = None  # No memory limit
    astore = ArrayStore(batch_limit=batch_limit, memory_limit=memory_limit)

    for i in range(100):
        astore.append(i)
    assert len(astore._arr) == batch_limit
    assert astore._arr == list(range(50, 100))
    assert (astore.get() == np.arange(50, 100)[:, None]).all() and astore.shape == (1,)

    msg = assert_error_msg(lambda: astore.append(np.array([10.0])))
    assert (
        msg
        == "Inhomogeneous types between stored values int64 and provided value float64."
    )

    astore = ArrayStore(batch_limit=batch_limit, memory_limit=memory_limit)
    assert astore.store_type is None
    astore.append(0)
    assert astore.store_type == int
    astore.reset()
    # reseting should not reset the type of shape
    msg = assert_error_msg(lambda: astore.append(0.0))
    assert (
        msg
        == "Inhomogeneous types between stored values int64 and provided value float64."
    )

    astore = ArrayStore(batch_limit=batch_limit, memory_limit=memory_limit)
    assert astore.get().shape == (0,)
    astore.append(0.0)
    assert astore.store_type == float
    assert astore.shape == (1,)
    astore.append(0.0)
    assert astore.get().shape == (2, 1)
    astore.append(0.0)
    assert astore.get().shape == (3, 1)
    astore = ArrayStore(batch_limit=batch_limit, memory_limit=memory_limit)
    assert len(astore) == 0
    astore.append(np.arange(10))
    assert astore.store_type == int
    astore.append(np.arange(10))
    astore.get()
    msg = assert_error_msg(lambda: astore.append(np.arange(5)))
    assert (
        msg
        == "Inhomogeneous shapes between stored values  (10,) and provided value (5,)"
    )
    assert astore.get().shape == (2, 10)
    astore.append(np.arange(10))
    assert astore.get().shape == (3, 10)

    astore.append(np.arange(10, 20))
    assert len(astore) == 4
    for i in range(3):
        astore.append(np.arange(10, 20))
    assert len(astore) == 4 + 3
    astore.get()
    assert len(astore) == 4 + 3

    assert (astore[-1] == np.arange(10, 20)).all()
    assert (astore[-5] == np.arange(10)).all()

    assert (astore[-len(astore)] == np.arange(10)).all()
    msg = assert_error_msg(lambda: astore[len(astore)])
    assert msg == "list index out of range"
    msg = assert_error_msg(lambda: astore[-len(astore) - 1])
    assert msg == "list index out of range"
    astore.reset()
    assert len(astore) == 0

    msg = assert_error_msg(lambda: astore[0])
    assert msg == "list index out of range"


def test_ma_limits():
    batch_limit = None
    memory_limit = None  # No memory limit
    array_store = ArrayStore(batch_limit=batch_limit, memory_limit=memory_limit)
    for i in range(100_000):
        array_store.append(i)
    assert len(array_store) == 100_000 and len(array_store._arr) == 100_000
    batch_limit = 1_000
    array_store = ArrayStore(batch_limit=batch_limit, memory_limit=memory_limit)
    for i in range(100_000):
        array_store.append(i)
    assert len(array_store) == batch_limit and len(array_store._arr) == batch_limit

    arr = np.random.rand(100, 100)
    memory_limit = 14 * sys.getsizeof(arr)
    batch_limit = 25
    array_store = ArrayStore(batch_limit=batch_limit, memory_limit=memory_limit)

    for i in range(100):
        arr = np.random.rand(100, 100)
        array_store.append(arr)
    practical_limit = ((14 // 10) + 1) * 10
    assert (
        len(array_store) == practical_limit and len(array_store._arr) == practical_limit
    )
    batch_limit = 19
    array_store = ArrayStore(batch_limit=batch_limit, memory_limit=memory_limit)

    for i in range(100):
        arr = np.random.rand(100, 100)
        array_store.append(arr)
    assert len(array_store) == batch_limit and len(array_store._arr) == batch_limit


def test_moving_average(assert_error_msg):
    ma = MovingAverage(batch_limit=100, memory_limit=100)
    assert np.isnan(ma.value)
    ma.append(np.array(0))
    ma.append(0)
    msg = assert_error_msg(lambda: ma.append(0.0))
    assert (
        msg
        == "Inhomogeneous types between stored values int64 and provided value float64."
    )
    ma.append(100)
    ma.append(100)
    assert ma == 50 and not ma != 50
    assert str(ma) == "5.00e+01"
    assert ma < 51 and ma > 49
    assert (ma.get() == np.array([0, 0, 100, 100])[:, None]).all()
    msg = assert_error_msg(lambda: ma.append(np.array([0, 1])))
    assert msg == "MovingAverage value must be scalar. Got [0 1]"
    msg = assert_error_msg(lambda: ma.append("t"))
    assert msg == "Invalid MovingAverage value type <class 'str'>"
    ma.reset()
    assert np.isnan(ma.value)
    assert str(ma) == "nan"


def my_eval_fn(*args, a1, a2="", **kwargs):
    return a1.mean()


def assert_store_unison_limits(n_labels, n_preds):
    labels = np.random.rand(1, n_labels)
    preds = np.random.rand(1, n_preds)
    labels_size = labels.size * labels.itemsize
    preds_size = preds.size * preds.itemsize
    bottleneck = np.random.randint(5, 30)
    mem_limit = max(labels_size, preds_size) * bottleneck
    store = PredictionStore(
        batch_limit=None,
        memory_limit=mem_limit,
        evaluation_functions=None,
    )
    store.append(preds=preds, labels=labels)
    assert store.limit is None
    # because mem limit is updated every 10 batches
    round_bottleneck = int(np.ceil(bottleneck / 10) * 10)
    for i in range(round_bottleneck - 1):
        store.append(
            preds=np.random.rand(1, n_preds), labels=np.random.rand(1, n_labels)
        )
    assert store.limit is None
    store.append(preds=np.random.rand(1, n_preds), labels=np.random.rand(1, n_labels))
    assert store.limit == round_bottleneck


def test_prediction_store(assert_error_msg):
    store = PredictionStore(
        batch_limit=None,
        memory_limit=10000,
        evaluation_functions=None,
    )
    assert store.evaluate() == {}
    msg = assert_error_msg(lambda: store.append())
    assert msg == "Must provide keyed batch arguments."
    msg = assert_error_msg(
        lambda: store.append(preds=np.array([4, 3, 0]), labels=np.array([5, 1, 1]))
    )
    assert (
        msg
        == "Missing batch dimension. If supplying a single value array, reshape to [B, 1] or if suppling a single a batch reshape to [1, N]."
    )
    preds = []
    labels = []
    for i in range(10):
        labels.append(np.random.rand(1, 200))
        preds.append(np.random.rand(1, 100))
        store.append(preds=preds[-1], labels=labels[-1])
    data = store.get()
    assert (data["preds"] == np.concatenate(preds)).all()
    assert (data["labels"] == np.concatenate(labels)).all()
    msg = assert_error_msg(
        lambda: store.append(
            preds=np.array([4, 3, 0]), labels=np.array([5, 1, 1]), xx=""
        )
    )
    assert (
        msg
        == "Inhomogeneous keys from the prediction store update. Expected: ['labels', 'preds'], received ['labels', 'preds', 'xx']"
    )
    msg = assert_error_msg(lambda: store.append(preds=np.array([4, 3, 0])))
    assert (
        msg
        == "Inhomogeneous keys from the prediction store update. Expected: ['labels', 'preds'], received ['preds']"
    )

    msg = assert_error_msg(
        lambda: store.append(
            preds=np.array([[4.0, 3, 0]]), labels=np.array([[5, 1, 1]])
        )
    )
    assert (
        msg
        == "Inhomogeneous shapes between stored values  (100,) and provided value (3,)"
    )
    msg = assert_error_msg(
        lambda: store.append(
            preds=np.array([np.arange(100)]), labels=np.array([np.arange(200)])
        )
    )
    assert (
        msg
        == "Inhomogeneous types between stored values float64 and provided value int64."
    )

    msg = assert_error_msg(
        lambda: store.append(
            preds=np.random.randn(10, 100), labels=np.random.randn(9, 200)
        )
    )
    assert (
        msg == "Inhomegenous batches between inputs. Sizes: {'preds': 10, 'labels': 9}"
    )
    assert store.evaluate() == {}
    store.reset()
    assert all([len(v) == 0 for k, v in store.get().items()])
    msg = assert_error_msg(
        lambda: store.append(
            preds=np.array([4, 3, 0]), labels=np.array([5, 1, 1]), xx=""
        )
    )
    assert (
        msg
        == "Inhomogeneous keys from the prediction store update. Expected: ['labels', 'preds'], received ['labels', 'preds', 'xx']"
    )
    msg = assert_error_msg(
        lambda: store.append(
            preds=np.array([[4.0, 3, 0]]), labels=np.array([[5, 1, 1]])
        )
    )
    assert (
        msg
        == "Inhomogeneous shapes between stored values  (100,) and provided value (3,)"
    )

    assert_store_unison_limits(100, 10)
    assert_store_unison_limits(10, 100)
    assert_store_unison_limits(10, 1000)
    assert_store_unison_limits(100, 10000)


def test_prediction_store_eval_fns(assert_error_msg):
    store = PredictionStore(
        batch_limit=None,
        memory_limit=10000,
        evaluation_functions={"accuracy": accuracy_score},
    )
    assert store.evaluate() == {}
    msg = assert_error_msg(
        lambda: store.append(preds=np.array([4, 3, 0]), labels=np.array([5, 1, 1]))
    )
    assert (
        "Batch keys do not match any function arguments: accuracy: ['y_true', 'y_pred', 'normalize', 'sample_weight']"
        == msg
    )
    store = PredictionStore(
        batch_limit=None,
        memory_limit=10000,
        evaluation_functions={"accuracy": accuracy_score},
    )
    store.append(y_true=np.random.randn(1, 100))
    msg = assert_error_msg(lambda: store.evaluate())
    assert msg == "missing a required argument: 'y_pred'"
    store = PredictionStore(
        batch_limit=200,
        memory_limit=10000,
        evaluation_functions={"accuracy": accuracy_score},
    )
    store.append(
        y_true=np.random.randint(2, size=(100, 1)),
        y_pred=np.random.randint(2, size=(100, 1)),
    )
    metrics = store.evaluate()
    assert metrics == store.evaluate()
    assert "accuracy" in metrics
    assert len(metrics) == 1
    store.append(y_true=np.ones((50, 1), dtype=int), y_pred=np.ones((50, 1), dtype=int))
    assert store.evaluate()["accuracy"] > metrics["accuracy"]
    store.append(
        y_true=np.ones((150, 1), dtype=int), y_pred=np.ones((150, 1), dtype=int)
    )
    assert store.evaluate()["accuracy"] == 1

    store = PredictionStore(
        batch_limit=None,
        memory_limit=10000,
        evaluation_functions=[accuracy_score, my_eval_fn],
    )
    assert "accuracy_score" in store.evaluation_function_arguments
    acc_args = store.evaluation_function_arguments["accuracy_score"]
    assert "y_pred" in acc_args and "y_true" in acc_args
    assert "my_eval_fn" in store.evaluation_function_arguments
    eval_args = store.evaluation_function_arguments["my_eval_fn"]
    assert "a1" in eval_args and "a2" in eval_args
    assert store.evaluate() == {}
    msg = assert_error_msg(
        lambda: store.append(
            y_true=np.ones((150, 1), dtype=int), y_pred=np.ones((150, 1), dtype=int)
        )
    )
    assert (
        "Batch keys do not match any function arguments: my_eval_fn: ['args', 'a1', 'a2', 'kwargs']"
        == msg
    )

    store = PredictionStore(
        batch_limit=300,
        memory_limit=10000,
        evaluation_functions=[accuracy_score, my_eval_fn],
    )
    store.append(
        y_true=np.ones((150, 1), dtype=int),
        y_pred=np.ones((150, 1), dtype=int),
        a1=np.ones((150, 1), dtype=int),
    )
    metrics = store.evaluate()
    assert metrics["accuracy_score"] == 1 and metrics["my_eval_fn"] == 1
    store.append(
        y_true=np.zeros((150, 1), dtype=int),
        y_pred=np.ones((150, 1), dtype=int),
        a1=np.zeros((150, 1), dtype=int),
    )

    metrics = store.evaluate()
    assert metrics["accuracy_score"] == 0.5 and metrics["my_eval_fn"] == 0.5
    store.append(
        y_true=np.zeros((75, 1), dtype=int),
        y_pred=np.ones((75, 1), dtype=int),
        a1=np.zeros((75, 1), dtype=int),
    )
    metrics = store.evaluate()
    assert metrics["accuracy_score"] == 0.25 and metrics["my_eval_fn"] == 0.25
    store.append(
        y_true=np.zeros((75, 1), dtype=int),
        y_pred=np.ones((75, 1), dtype=int),
        a1=np.zeros((75, 1), dtype=int),
    )
    metrics = store.evaluate()
    assert metrics["accuracy_score"] == 0 and metrics["my_eval_fn"] == 0


if __name__ == "__main__":
    from tests.conftest import _assert_error_msg

    # test_array_store(_assert_error_msg)
    # test_ma_limits()
    # test_moving_average(_assert_error_msg)
    # test_prediction_store(_assert_error_msg)
    test_prediction_store_eval_fns(_assert_error_msg)
