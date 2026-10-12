from typing import TYPE_CHECKING

try:
    import pkg.aliased
except ImportError:
    pkg = None

if TYPE_CHECKING:
    from pkg.core import Engine


def make() -> "Engine":
    return pkg.aliased.build()
