'''
Description: 
version: 
Author: huangzg
LastEditors: huangzg
Date: 2023-07-28 22:56:02
LastEditTime: 2023-08-26 14:12:11
'''
import logging
import os
from packaging import version
import importlib
import warnings
from typing import Union, List, Tuple, Dict

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # set logger level

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True


def get_common_args(methods, input_args:dict,exclude_keys:Union[str,list,tuple,set]=None):
        """in case of args conflict, use their common args as a candidate"""
        import inspect
        common_kwargs_keys = set(inspect.getfullargspec(methods).args)&set(input_args.keys())
        common_kwargs = {key:input_args[key] for key in common_kwargs_keys}
        if exclude_keys:
            if isinstance(exclude_keys,str):
                common_kwargs.pop(exclude_keys)
            else:
                for key in exclude_keys:
                    common_kwargs.pop(key)
             
        return common_kwargs