import pytest


@pytest.fixture
def assert_error_msg():
    def eval_fn(fn, error_msg):
        try:
            fn()
            assert False
        except Exception as excp:
            if not error_msg == str(excp):
                raise excp

    return eval_fn

# Common function written for tests to compare the error messages
