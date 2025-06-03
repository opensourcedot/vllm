import logging
import importlib
import sys
import types
from typing import Optional

from vllm.plugins import load_plugins_by_group


logger = logging.getLogger(__name__)


_current_framework = None


def set_torch_framework() -> None:
    import torch
    global _current_framework
    _current_framework = torch
    sys.modules[f'{__name__}.distributed'] = torch.distributed
    sys.modules[f'{__name__}.nn'] = torch.nn


def resolve_framework_by_name(name: str) -> types.ModuleType:
    framework_plugins = load_plugins_by_group('vllm.framework_plugins')

    if name in framework_plugins.keys():
        assert callable(framework_plugins[name])
        framework = framework_plugins[name]()
    else:
        raise RuntimeError(
            f"No framework plugin named {name} is found!")
    return framework


def set_current_framework(name: str = "pt") -> None:
    """Set the current framework to the given framework."""
    global _current_framework
    if name != "pt":
        framework = resolve_framework_by_name(name)
        _current_framework = framework
        importlib.import_module(framework)
    else:
        set_torch_framework()
    logger.info(f"Current framework set to {framework.__name__}")


def __getattr__(name: str):
    if name == 'current_framework':
        global _current_framework
        if _current_framework is None:
            set_torch_framework()
        return _current_framework
    else:
        raise AttributeError(
            f"No attribute named '{name}' exists in {__name__}.")
