import importlib
import pytest
from sklplus._coverage import PUBLIC_SYMBOLS


def test_version():
    import sklplus
    assert sklplus.__version__ == "0.1.2"


@pytest.mark.parametrize("module_path,attr", PUBLIC_SYMBOLS)
def test_public_symbol_importable(module_path, attr):
    mod = importlib.import_module(module_path)
    assert hasattr(mod, attr), f"{module_path} missing {attr}"


def test_no_get_model():
    import sklplus

    assert not hasattr(sklplus, "get_model")
