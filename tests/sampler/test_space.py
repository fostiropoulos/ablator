import copy
import pickle
import pytest
from ablator.config.hpo import SearchSpace

sub_spaces = [
    {
        "value_range": (0, 1),
        "value_type": "float",
    },
    {"sub_configuration": {"name": "sgd", "arguments": {"lr": 0.1}}},
    {
        "sub_configuration": {
            "name": "adam",
            "arguments": {
                "lr": {
                    "value_range": (0, 1),
                    "value_type": "float",
                },
                "wd": 0.9,
            },
        }
    },
    {
        "sub_configuration": {
            "name": "adam",
            "arguments": {
                "lr": {
                    "subspaces": [
                        {
                            "value_range": (0, 1),
                            "value_type": "float",
                        },
                        {
                            "value_range": (0, 1),
                            "value_type": "float",
                        },
                    ]
                },
                "wd": 0.9,
            },
        }
    },
]
_optim_config = SearchSpace(subspaces=sub_spaces)


@pytest.fixture()
def optim_config():
    return copy.deepcopy(_optim_config)


def _parse_type(a, b):
    _a = a.__dict__ if hasattr(a, "__dict__") else a
    _b = b.__dict__ if hasattr(b, "__dict__") else b
    return _a, _b


def _assert_nested_equal(reference, sample):
    if not isinstance(reference, dict):
        assert reference == sample
        return
    for k, v in reference.items():
        if k == "value_range" and v is not None:
            assert sample[k] == (type(sample[k])(str(_v) for _v in v))
        elif isinstance(v, dict):
            _assert_nested_equal(v, sample[k])
        elif isinstance(v, list):
            for i, _v in enumerate(v):
                a, b = _parse_type(_v, sample[k][i])
                _assert_nested_equal(a, b)
        elif isinstance(type(v), type):
            a, b = _parse_type(v, sample[k])
            _assert_nested_equal(a, b)
        else:
            assert sample[k] == v


def test_search_space(optim_config: SearchSpace):
    space = {
        "train_config.optimizer_config": optim_config,
        "b": SearchSpace(value_range=(-10, 10), value_type="float"),
    }
    lr_sp = (
        space["train_config.optimizer_config"]
        .subspaces[2]
        .sub_configuration["arguments"]["lr"]
    )
    assert (
        isinstance(
            lr_sp,
            SearchSpace,
        )
        and lr_sp.value_range
        == ["0", "1"]  # this is because we cast to str for float safety
        and lr_sp.categorical_values is None
    )
    assert isinstance(
        space["train_config.optimizer_config"]
        .subspaces[1]
        .sub_configuration["arguments"]["lr"],
        float,
    )
    assert (
        space["train_config.optimizer_config"]
        .subspaces[2]
        .sub_configuration["arguments"]["wd"]
        == 0.9
    )
    optim_config.make_dict(space["train_config.optimizer_config"].annotations)
    converted_subspaces = optim_config.to_dict()["subspaces"]

    for i, _subspace in enumerate(sub_spaces):
        _assert_nested_equal(_subspace, converted_subspaces[i])


def test_search_space_paths(optim_config: SearchSpace):
    paths = optim_config.make_paths()
    assert sorted(paths) == ["", "arguments.lr", "arguments.wd", "name"]


def test_copy(optim_config: SearchSpace):
    _optim_config = copy.deepcopy(optim_config)
    _assert_nested_equal(_optim_config.__dict__, optim_config.__dict__)
    _optim_config = copy.copy(optim_config)
    _assert_nested_equal(_optim_config.__dict__, optim_config.__dict__)


def test_pickle(optim_config: SearchSpace):
    pickled_config = pickle.loads(pickle.dumps(optim_config))
    _assert_nested_equal(pickled_config.__dict__, optim_config.__dict__)


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]
    kwargs = {"optim_config": copy.deepcopy(_optim_config)}

    run_tests_local(test_fns, kwargs)
