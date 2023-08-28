from ablator.config.utils import flatten_nested_dict


def test_flatten_nested_dict():
    out = flatten_nested_dict({"a": {"b": 1, "c": {"d": 2}}, "e": [3, 4]}, False, "#")
    assert out == {"e": [3, 4], "a#b": 1, "a#c#d": 2}
    out = flatten_nested_dict({"a": {"b": 1, "c": {"d": 2}}, "e": [3, 4]}, True, "#")
    assert out == {"a#b": 1, "e#0": 3, "e#1": 4, "a#c#d": 2}


if __name__ == "__main__":
    test_flatten_nested_dict()
