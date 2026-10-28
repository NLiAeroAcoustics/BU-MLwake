"""Utility functions for loading, stacking, splitting, and normalizing
transformer-style turbofan datasets.

This module is written for the newer database structure produced by
Data_preprocessing_supaero_transformer.ipynb, where X, Y, Y_scalar, and label
are dictionaries keyed by variable names.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = np.ndarray
KeySpec = Union[str, Sequence[str]]

# Default field ordering used by the legacy CNN baseline.
DEFAULT_OUTPUT_KEYS = ("tke", "LenScale", "vx", "vt", "p", "T", "v_mag")
DEFAULT_COORD_KEYS = ("x", "y", "z", "r", "th")
DEFAULT_SCALAR_KEYS = ("torque", "pr", "tr")

# Backward-compatible map from old channel indices to new dictionary keys.
LEGACY_Y_INDEX_MAP = {
    0: "tke",
    1: "omega",
    2: "LenScale",
    3: "vx",
    4: "vt",
    5: "p",
    6: "T",
    7: "v_mag",
    8: "rho",
    9: "x",
    10: "y",
    11: "z",
    12: "r",
    13: "th",
}


class DatasetFormatError(ValueError):
    """Raised when a database does not match the expected dictionary format."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_pickle(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def _as_float32(array: ArrayLike) -> ArrayLike:
    return np.asarray(array, dtype=np.float32)


def _as_2d(array: ArrayLike, key: str) -> ArrayLike:
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise DatasetFormatError(f"Field '{key}' must be 1D or 2D, got shape {arr.shape}.")
    return arr


def _normalize_key_spec(keys: Union[None, int, str, Sequence[Union[int, str]]], *, mapping: Mapping[int, str]) -> List[str]:
    if keys is None:
        return []
    if isinstance(keys, (int, str)):
        keys = [keys]

    resolved: List[str] = []
    for key in keys:
        if isinstance(key, str):
            resolved.append(key)
        else:
            if key not in mapping:
                raise KeyError(f"Unknown legacy channel index {key}. Available indices: {sorted(mapping)}")
            resolved.append(mapping[key])
    return resolved


def _require_dict_dataset(dat: Mapping[str, Any]) -> None:
    required_top = ("X", "Y", "Y_scalar", "label")
    missing = [k for k in required_top if k not in dat]
    if missing:
        raise DatasetFormatError(f"Missing top-level keys: {missing}")
    for key in required_top:
        if not isinstance(dat[key], Mapping):
            raise DatasetFormatError(
                f"Expected dat['{key}'] to be a dictionary for the transformer dataset, got {type(dat[key]).__name__}."
            )


# ---------------------------------------------------------------------------
# Public loading / stacking helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Union[str, Path]) -> Dict[str, Any]:
    """Load the transformer-style dataset and validate its basic structure."""
    dat = _load_pickle(path)
    _require_dict_dataset(dat)
    return dat


def available_keys(path: Union[str, Path]) -> Dict[str, List[str]]:
    """Return available keys in X, Y, Y_scalar, and label."""
    dat = load_dataset(path)
    return {name: list(dat[name].keys()) for name in ("X", "Y", "Y_scalar", "label")}


def stack_input_features(dat: Mapping[str, Any], input_keys: Optional[Sequence[str]] = None) -> ArrayLike:
    """Stack selected input feature blocks into a single 2D array [N, n_features]."""
    x_dict = dat["X"]
    if input_keys is None:
        input_keys = list(x_dict.keys())

    blocks = []
    n_samples = None
    for key in input_keys:
        if key not in x_dict:
            raise KeyError(f"Input key '{key}' not found. Available keys: {list(x_dict.keys())}")
        arr = _as_2d(x_dict[key], key)
        if n_samples is None:
            n_samples = arr.shape[0]
        elif arr.shape[0] != n_samples:
            raise DatasetFormatError(f"Input field '{key}' has {arr.shape[0]} samples, expected {n_samples}.")
        blocks.append(arr)

    if not blocks:
        raise ValueError("No input keys were selected.")
    return _as_float32(np.concatenate(blocks, axis=1))


def stack_output_fields(dat: Mapping[str, Any], output_keys: Optional[Sequence[str]] = None) -> ArrayLike:
    """Stack selected 2D output fields into [N, H, W, C]."""
    y_dict = dat["Y"]
    if output_keys is None:
        output_keys = list(DEFAULT_OUTPUT_KEYS)

    blocks = []
    ref_shape = None
    for key in output_keys:
        if key not in y_dict:
            raise KeyError(f"Output key '{key}' not found. Available keys: {list(y_dict.keys())}")
        arr = np.asarray(y_dict[key])
        if arr.ndim != 3:
            raise DatasetFormatError(f"Output field '{key}' must have shape [N, H, W], got {arr.shape}.")
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise DatasetFormatError(f"Output field '{key}' has shape {arr.shape}, expected {ref_shape}.")
        blocks.append(arr[..., None])

    return _as_float32(np.concatenate(blocks, axis=-1))


def stack_coordinate_fields(dat: Mapping[str, Any], coord_keys: Optional[Sequence[str]] = None) -> ArrayLike:
    """Stack selected coordinate fields into [N, H, W, Ccoord]."""
    return stack_output_fields({"Y": dat["Y"]}, output_keys=coord_keys or DEFAULT_COORD_KEYS)


def get_label_dict(dat: Mapping[str, Any]) -> Dict[str, ArrayLike]:
    """Return labels as a plain dict of numpy arrays."""
    return {k: np.asarray(v) for k, v in dat["label"].items()}


# ---------------------------------------------------------------------------
# Backward-compatible loaders used by the legacy CNN notebook
# ---------------------------------------------------------------------------

def load_data(path: Union[str, Path]):
    """Return the raw dict-style components from the new database."""
    dat = load_dataset(path)
    return dat["X"], dat["Y"], dat["Y_scalar"], get_label_dict(dat)


def load_data_multiChan(
    file,
    input_keys,
    output_keys=("tke", "LenScale", "vx", "vt", "p", "T", "v_mag"),
    coord_keys=("x", "y", "z", "r", "th"),total=False,metric=True):
    import pickle
    import numpy as np

    with open(file, "rb") as f:
        dat = pickle.load(f)
    if metric is True:
        rg = 287.05
    else:
        rg = 1716
    if total is True:
        M = dat['Y']["v_mag"]/(1.4*rg*dat['Y']["T"])**0.5
        dat['Y']["T"] = dat['Y']["T"]*(1+0.4/2*M**2)
        dat['Y']["p"] = dat['Y']["p"]*((1+0.4/2*M**2)**(-1.4/0.4))**(-1)        
    # ----- stack selected X inputs -----
    x_list = []
    for key in input_keys:
        arr = np.asarray(dat["X"][key])
        arr = arr.reshape(len(arr), -1)
        x_list.append(arr)
    input_ = np.concatenate(x_list, axis=1)

    # ----- stack selected Y outputs -----
    y_list = []
    for key in output_keys:
        arr = np.asarray(dat["Y"][key])[..., np.newaxis]
        y_list.append(arr)
    output_ = np.concatenate(y_list, axis=-1)

    # ----- stack coordinates -----
    c_list = []
    for key in coord_keys:
        arr = np.asarray(dat["Y"][key])[..., np.newaxis]
        c_list.append(arr)
    coordinate = np.concatenate(c_list, axis=-1)

    label = dat["label"]

    print("input shape      :", input_.shape)
    print("output shape     :", output_.shape)
    print("coordinate shape :", coordinate.shape)

    return input_, output_, coordinate, label


def load_data_scalar(
    path: Union[str, Path],
    scalar_keys: Sequence[str] = DEFAULT_SCALAR_KEYS,
    *,
    input_keys: Optional[Sequence[str]] = None,
):
    """Load selected scalar outputs and stacked input features."""
    dat = load_dataset(path)
    y_scalar = dat["Y_scalar"]
    label = get_label_dict(dat)
    input_ = stack_input_features(dat, input_keys=input_keys)

    outputs = []
    for key in scalar_keys:
        if key not in y_scalar:
            raise KeyError(f"Scalar output key '{key}' not found. Available keys: {list(y_scalar.keys())}")
        outputs.append(np.asarray(y_scalar[key]).reshape(-1))

    if len(outputs) == 1:
        return outputs[0], label, input_
    return (*outputs, label, input_)


# ---------------------------------------------------------------------------
# Selection / splitting helpers
# ---------------------------------------------------------------------------

def _label_column(label: Union[Mapping[str, Any], ArrayLike], key: str) -> ArrayLike:
    if isinstance(label, Mapping):
        if key not in label:
            raise KeyError(f"Label key '{key}' not found. Available keys: {list(label.keys())}")
        arr = np.asarray(label[key]).reshape(-1)
        return arr
    raise DatasetFormatError("This rewritten prelim.py expects label to be a dictionary of arrays.")


def find_slice_by_geom(
    label: Mapping[str, Any],
    geomlist: Sequence[Union[int, str]],
    x_ind: Optional[Union[int, Sequence[int]]] = None,
    *,
    geom_key: str = "geometry number",
    slice_key: str = "Slice index",
) -> List[int]:
    """Find sample indices for selected geometries and optional slice indices."""
    geom_arr = _label_column(label, geom_key)
    geom_arr_str = np.asarray([str(int(g)) if isinstance(g, (np.integer, int, float)) and float(g).is_integer() else str(g) for g in geom_arr])
    geom_targets = {str(g) for g in geomlist}

    mask = np.isin(geom_arr_str, list(geom_targets))

    if x_ind is not None:
        slice_arr = _label_column(label, slice_key).astype(int)
        if isinstance(x_ind, Sequence) and not isinstance(x_ind, (str, bytes)):
            x_targets = {int(v) for v in x_ind}
            mask &= np.isin(slice_arr, list(x_targets))
        else:
            mask &= (slice_arr == int(x_ind))

    return np.flatnonzero(mask).tolist()

def find_slice_by_geom_and_slice(label, geom_values=None, rpm_values = None, slice_values=None):
    """
    Return sample indices whose labels match:
      - label["geometry number"] in geom_values (if provided)
      - label["Slice index"] in slice_values (if provided)

    If geom_values is None, all geometries are allowed.
    If slice_values is None, all slice indices are allowed.

    Parameters
    ----------
    label : dict
        Label dictionary containing:
            label["geometry number"]
            label["Slice index"]

    geom_values : list, array-like, or None
        Allowed geometry-number values.

    slice_values : list, array-like, or None
        Allowed slice-index values.

    Returns
    -------
    matched_indices : list
        List of sample indices satisfying the requested conditions.
    """

    import numpy as np

    geom_arr = np.asarray(label["geometry number"]).reshape(-1)
    slice_arr = np.asarray(label["Slice index"]).reshape(-1)
    rpm_arr = np.asarray(label["rpm"]).reshape(-1)

    if geom_values is not None:
        geom_set = set(geom_values)

    if slice_values is not None:
        slice_set = set(slice_values)
    if rpm_values is not None:
        rpm_set = set(rpm_values)
        
    matched_indices = []
    for i, (g, s, r) in enumerate(zip(geom_arr, slice_arr, rpm_arr)):
        geom_ok = True if geom_values is None else (g in geom_set)
        slice_ok = True if slice_values is None else (s in slice_set)
        rpm_ok = True if rpm_values is None else (r in rpm_set)

        if geom_ok and slice_ok and rpm_ok:
            matched_indices.append(i)

    return matched_indices

def drop_indices(total_size: int, indices_to_drop: Sequence[int]) -> List[int]:
    mask = np.ones(total_size, dtype=bool)
    mask[np.asarray(indices_to_drop, dtype=int)] = False
    return np.flatnonzero(mask).tolist()

### transformer paring

def make_training_pairs(input_, output_, frac=0.10, seed=42, return_indices=False):
    """
    Randomly select a fraction of all possible sample pairs, then build:
      - reference output
      - target output
      - delta input = input_target - input_reference

    This is meant to be used BEFORE regularization.

    Parameters
    ----------
    input_ : np.ndarray
        Shape (N, n_features)

    output_ : np.ndarray
        Shape (N, H, W, C)

    frac : float, optional
        Fraction of all possible pairs to keep. Example: 0.10 means 10%.

    seed : int, optional
        Random seed for reproducibility.

    return_indices : bool, optional
        If True, also return the selected pair indices.

    Returns
    -------
    output_ref : np.ndarray
        Reference flow fields, shape (N_pairs, H, W, C)

    delta_input : np.ndarray
        Input differences, shape (N_pairs, n_features)

    output_tar : np.ndarray
        Target flow fields, shape (N_pairs, H, W, C)

    pair_idx : np.ndarray, optional
        Shape (N_pairs, 2), only returned if return_indices=True
    """
    import numpy as np
    from itertools import combinations

    n_samples = input_.shape[0]

    if output_.shape[0] != n_samples:
        raise ValueError("input_ and output_ must have the same number of samples.")

    all_pairs = list(combinations(range(n_samples), 2))
    n_all_pairs = len(all_pairs)

    n_keep = max(1, int(frac * n_all_pairs))

    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_all_pairs, size=n_keep, replace=False)

    pair_idx = np.array([all_pairs[i] for i in chosen], dtype=int)

    ref_idx = pair_idx[:, 0]
    tar_idx = pair_idx[:, 1]

    output_ref = output_[ref_idx]
    output_tar = output_[tar_idx]
    delta_input = input_[tar_idx] - input_[ref_idx]

    if return_indices:
        return output_ref, delta_input, output_tar, pair_idx
    else:
        return output_ref, delta_input, output_tar
# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _compute_input_norm(input_: ArrayLike, fit_indices: Sequence[int], epsi: float) -> Dict[str, ArrayLike]:
    fit = input_[fit_indices]
    return {
        "input_mean": fit.mean(axis=0),
        "input_std": fit.std(axis=0),
        "epsi": float(epsi),
    }


def _compute_output_norm(output_: ArrayLike, fit_indices: Sequence[int]) -> Dict[str, ArrayLike]:
    fit = output_[fit_indices]
    return {
        "output_mean": fit.mean(axis=(0, 1, 2)),
        "output_std": fit.std(axis=(0, 1, 2)),
    }


def _save_norm(norm: Mapping[str, Any], norm_path: Optional[Union[str, Path]]) -> None:
    if norm_path is None:
        return
    Path(norm_path).parent.mkdir(parents=True, exist_ok=True)
    with open(norm_path, "wb") as fp:
        pickle.dump(dict(norm), fp)


def regularized_train_test(
    input_: ArrayLike,
    var: ArrayLike,
    evalSlice: Sequence[int],
    trainSlice: Optional[Sequence[int]] = None,
    nrange: Optional[Sequence[int]] = None,
    *,
    norm_path: Optional[Union[str, Path]] = "normalize_parameters.pkl",
    epsi: float = 1.0e-6,
):
    """Normalize inputs using the training subset, then split train/test."""
    n_samples = input_.shape[0]
    eval_idx = np.asarray(evalSlice, dtype=int)

    if trainSlice is None:
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[eval_idx] = False
        train_idx = np.flatnonzero(train_mask)
    else:
        train_idx = np.asarray(trainSlice, dtype=int)

    fit_idx = np.asarray(nrange if nrange is not None else train_idx, dtype=int)
    norm = _compute_input_norm(input_, fit_idx, epsi)
    input_reg = (input_ - norm["input_mean"]) / (norm["input_std"] + epsi)

    x_train = input_reg[train_idx]
    y_train = var[train_idx]
    x_test = input_reg[eval_idx]
    y_test = var[eval_idx]

    _save_norm(norm, norm_path)
    return x_train, x_test, y_train, y_test


def regularized_train_test_mc(
    input_: ArrayLike,
    output_: ArrayLike,
    evalSlice: Sequence[int],
    trainSlice: Optional[Sequence[int]] = None,
    nrange: Optional[Sequence[int]] = None,
    *,
    norm_path: Optional[Union[str, Path]] = "normalize_parameters.pkl",
    epsi: float = 1.0e-6,
):
    """Normalize inputs and multi-channel outputs using the training subset."""
    n_samples = input_.shape[0]
    eval_idx = np.asarray(evalSlice, dtype=int)

    if trainSlice is None:
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[eval_idx] = False
        train_idx = np.flatnonzero(train_mask)
    else:
        train_idx = np.asarray(trainSlice, dtype=int)

    fit_idx = np.asarray(nrange if nrange is not None else train_idx, dtype=int)
    norm = _compute_input_norm(input_, fit_idx, epsi)
    norm.update(_compute_output_norm(output_, fit_idx))

    input_reg = (input_ - norm["input_mean"]) / (norm["input_std"] + epsi)
    output_reg = (output_ - norm["output_mean"]) / (norm["output_std"] + epsi)

    x_train = input_reg[train_idx]
    y_train = output_reg[train_idx]
    x_test = input_reg[eval_idx]
    y_test = output_reg[eval_idx]

    _save_norm(norm, norm_path)
    return x_train, x_test, y_train, y_test


def regularized_train_test_renorm(
    input_: ArrayLike,
    var: ArrayLike,
    evalSlice: Sequence[int],
    trainSlice: Optional[Sequence[int]] = None,
    *,
    nfile: Union[str, Path] = "normalize_parameters.pkl",
):
    """Apply previously saved normalization parameters to a new split."""
    with open(nfile, "rb") as fp:
        norm = pickle.load(fp)

    epsi = norm["epsi"]
    input_reg = (input_ - norm["input_mean"]) / (norm["input_std"] + epsi)

    n_samples = input_.shape[0]
    eval_idx = np.asarray(evalSlice, dtype=int)
    if trainSlice is None:
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[eval_idx] = False
        train_idx = np.flatnonzero(train_mask)
    else:
        train_idx = np.asarray(trainSlice, dtype=int)

    x_train = input_reg[train_idx]
    y_train = var[train_idx]
    x_test = input_reg[eval_idx]
    y_test = var[eval_idx]
    return x_train, x_test, y_train, y_test


# Backward-compatible alias for your old typo-ed function name.
def regularized_train_test_renrom(*args, **kwargs):
    return regularized_train_test_renorm(*args, **kwargs)


def regularize(input_: ArrayLike, nfile: Union[str, Path]) -> ArrayLike:
    with open(nfile, "rb") as fp:
        norm = pickle.load(fp)
    epsi = norm["epsi"]
    return (input_ - norm["input_mean"]) / (norm["input_std"] + epsi)


# ---------------------------------------------------------------------------
# Metadata splitting helpers
# ---------------------------------------------------------------------------

def _slice_dict(data: Mapping[str, ArrayLike], idx: Sequence[int]) -> Dict[str, ArrayLike]:
    return {k: np.asarray(v)[idx] for k, v in data.items()}


def split_coordinate_label(
    coordinate: ArrayLike,
    label: Mapping[str, ArrayLike],
    evalSlice: Sequence[int],
    trainSlice: Optional[Sequence[int]] = None,
):
    """Split stacked coordinates and label dictionaries into train/test parts."""
    n_samples = coordinate.shape[0]
    eval_idx = np.asarray(evalSlice, dtype=int)

    if trainSlice is None:
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[eval_idx] = False
        train_idx = np.flatnonzero(train_mask)
    else:
        train_idx = np.asarray(trainSlice, dtype=int)

    label_train = _slice_dict(label, train_idx)
    coordinate_train = coordinate[train_idx]
    label_test = _slice_dict(label, eval_idx)
    coordinate_test = coordinate[eval_idx]
    return label_train, coordinate_train, label_test, coordinate_test
