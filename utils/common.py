from typing import Mapping, Any, Tuple, Callable
import importlib
import os
from urllib.parse import urlparse

import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np

# from torch.hub import download_url_to_file, get_dir


def get_obj_from_str(string: str, reload: bool=False) -> Any:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))