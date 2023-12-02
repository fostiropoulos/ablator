from pathlib import Path
import pytest
import torch
import os
from ablator.utils.file import default_val_parser, nested_set, truncate_utf8_chars


def test_default_val_parser_tensor():
    tensor = torch.tensor([1, 2, 3])
    expected_output = [1, 2, 3]
    # Test `default_val_parser` with tensor input to check if it convert to list
    assert default_val_parser(tensor) == expected_output


def test_nested_set_path_not_exist():
    dict_ = {"a": 1}
    keys = ["b", "c"]
    value = 2
    expected_output = {"a": 1, "b": {"c": 2}}
    # Test `nested_set` with keys not in dict to check if it add the keys and value to the dict
    assert nested_set(dict_, keys, value) == expected_output


def test_truncate_utf8_chars_not_found(tmp_path: Path):
    file_name = tmp_path / "test_file.txt"
    last_char = "Z"  # This char will not be in the file.
    with open(file_name, "w") as f:
        f.write("abc")

    with pytest.raises(RuntimeError) as err:
        truncate_utf8_chars(file_name, last_char)

    # Test `truncate_utf8_chars` with last_char not in that file to check if it raise an error
    assert "Could not truncate" in str(err.value)
    os.remove(file_name)


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]

    run_tests_local(test_fns)
