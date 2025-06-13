import logging
from typing import TYPE_CHECKING, Optional
from itertools import chain

from vllm.plugins import load_plugins_by_group
from vllm.frameworks.framework import Framework


logger = logging.getLogger(__name__)


def pytorch_framework_plugin() -> Optional[Framework]:
    from .pytorch import PyTorchFramework  # noqa: F401
    logger.debug("Confirmed pytorch framework is available.")
    return PyTorchFramework()


builtin_framework_plugins = {
    'pt': pytorch_framework_plugin,
}


def resolve_current_framework_by_name(name="pt") -> Optional[Framework]:
    framework_plugins = load_plugins_by_group('vllm.framework_plugins')

    if name in builtin_framework_plugins.keys():
        framework = builtin_framework_plugins[name]()
    elif name in framework_plugins.keys():
        assert callable(framework_plugins[name])
        framework = framework_plugins[name]()
    else:
        raise RuntimeError(
            f"No framework plugin named {name} is found!")
    return framework


_current_framework = None

if TYPE_CHECKING:
    current_framework: Framework


def __getattr__(name: str):
    if name == 'current_framework':
        global _current_framework
        if _current_framework is None:
            _current_framework = resolve_current_framework_by_name()
        return _current_framework
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(
            f"No attribute named '{name}' exists in {__name__}.")


__all__ = [
    'Framework', 'current_framework',
]