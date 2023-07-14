import pytest
import torch
import os
from ablator.utils.file import (default_val_parser, nested_set, truncate_utf8_chars)


def test_default_val_parser_tensor():
    tensor = torch.tensor([1, 2, 3])
    expected_output = [1, 2, 3]
    # test `default_val_parser` with tensor input
    assert default_val_parser(tensor) == expected_output


def test_nested_set_path_not_exist():
    dict_ = {'a': 1}
    keys = ['b', 'c']
    value = 2
    expected_output = {'a': 1, 'b': {'c': 2}}
    # test `nested_set` with keys not in dict
    assert nested_set(dict_, keys, value) == expected_output


def test_truncate_utf8_chars_not_found():
    file_name = 'test_file.txt'
    last_char = 'Z'  # This char will not be in the file.
    with open(file_name, 'w') as f:
        f.write('abc')

    with pytest.raises(RuntimeError) as err:
        truncate_utf8_chars(file_name, last_char)

    # test `truncate_utf8_chars` with last_char not in that file
    assert 'Could not truncate' in str(err.value)
    os.remove(file_name)
