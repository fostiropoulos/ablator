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
optim_config = SearchSpace(subspaces=sub_spaces)


def test_search_space():
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

    def _assert_nested_equal(reference, sample):
        if not isinstance(reference, dict):
            assert reference == sample
        for k, v in reference.items():
            if k == "value_range" and v is not None:
                assert sample[k] == tuple(str(_v) for _v in v)
            elif isinstance(v, dict):
                _assert_nested_equal(v, sample[k])
            elif isinstance(v, list):
                [_assert_nested_equal(_v, sample[k][i]) for i, _v in enumerate(v)]
            else:
                assert sample[k] == v

    for i, _subspace in enumerate(sub_spaces):
        _assert_nested_equal(_subspace, converted_subspaces[i])


def test_search_space_paths():
    paths = optim_config.make_paths()
    assert sorted(paths) == ["", "arguments.lr", "arguments.wd", "name"]


if __name__ == "__main__":
    test_search_space()
    test_search_space_paths()
