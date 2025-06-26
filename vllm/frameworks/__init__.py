import os
import logging
import importlib
import sys
import types
from typing import Optional

from vllm.plugins import load_plugins_by_group


logger = logging.getLogger(__name__)


_current_framework = None



def resolve_framework_by_name(name: str) -> types.ModuleType:
    framework_plugins = load_plugins_by_group('vllm.framework_plugins')

    if name in framework_plugins.keys():
        assert callable(framework_plugins[name])
        framework = framework_plugins[name]()
    else:
        raise RuntimeError(
            f"No framework plugin named {name} is found!")
    return framework


def set_current_framework() -> None:
    """Set the current framework to the given framework."""
    global _current_framework
    framework_name = os.environ.get('VLLM_FRAMEWORK')
    if framework_name is not None:
        framework = resolve_framework_by_name(framework_name)
        _current_framework = framework
        logger.info(f"Current framework set to {framework_name}")
    else:
        import torch
        import torch.library
        import torch.distributed
        import torch.nn
        _current_framework = torch
        logger.info("Current framework set to pt")

    sys.modules[f'{__name__}.distributed'] = _current_framework.distributed
    sys.modules[f'{__name__}.library'] = _current_framework.library
    sys.modules[f'{__name__}.nn'] = _current_framework.nn


def __getattr__(name: str):
    if name == 'current_framework':
        global _current_framework
        if _current_framework is None:
            set_current_framework()
        return _current_framework
    else:
        raise AttributeError(
            f"No attribute named '{name}' exists in {__name__}.")
