import numpy as np
import cupy as cp


def save_params_dict_cupy(params, path):
    """
    Save a dictionary of parameters to a file in cupy format.
    :param params: Parameters to save.
    :param path: Path to save the dictionary to.
    :return:
    """
    params_dict_cupy = params.copy()
    for k, v in params['model'].items():
        params_dict_cupy['model'][k] = cp.asarray(v)
    save_params_dict(params_dict_cupy, path)


def load_params_dict_cupy(path):
    """
    Load a dictionary of parameters from a file in cupy format.
    :param path: Path to save the dictionary to.
    :return:
    """
    params = load_params_dict(path)
    for k, v in params['model'].items():
        params['model'][k] = cp.asarray(v)
    return params


def save_params_dict(params, path):
    """
    Save a dictionary of parameters to a file.
    :param params: A dictionary of parameters.
    :param path:
    :return:
    """
    np.save(path, params)


def load_params_dict(path) -> dict:
    """
    Load a dictionary of parameters from a file.
    :param path:
    :return: A dictionary of parameters.
    """
    params = np.load(path, allow_pickle=True)
    return params
