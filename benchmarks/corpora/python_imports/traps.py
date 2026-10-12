"""Not static imports: none of these create a module dependency edge."""
import importlib

# import pkg.util
EXAMPLE = "from pkg import core"


def load(name="pkg.core"):
    return importlib.import_module(name)
