import importlib
import pytest
from sklearnplus._coverage import PUBLIC_SYMBOLS


def test_version():
    import sklearnplus
    assert sklearnplus.__version__ == "0.1.0"


@pytest.mark.parametrize("module_path,attr", PUBLIC_SYMBOLS)
def test_public_symbol_importable(module_path, attr):
    mod = importlib.import_module(module_path)
    assert hasattr(mod, attr), f"{module_path} missing {attr}"
