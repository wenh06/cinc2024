"""
Miscellaneous functions.
"""

import os
import re
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
import requests
import yaml

__all__ = [
    "func_indicator",
    "predict_proba_ordered",
    "url_is_reachable",
    "load_submission_log",
    "get_record_list_recursive3",
]


def func_indicator(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print("\n" + "-" * 100)
            print(f"  Start {name}  ".center(100, "-"))
            print("-" * 100 + "\n")
            func(*args, **kwargs)
            print("\n" + "-" * 100)
            print(f"  End {name}  ".center(100, "-"))
            print("-" * 100 + "\n")

        return wrapper

    return decorator


def predict_proba_ordered(probs: np.ndarray, classes_: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    """Workaround for the problem that one can not set explicitly
    the list of classes for models in sklearn.

    Modified from https://stackoverflow.com/a/32191708

    Parameters
    ----------
    probs : numpy.ndarray
        Array of probabilities, output of `predict_proba` method of sklearn models.
    classes_ : numpy.ndarray
        Array of classes, output of `classes_` attribute of sklearn models.
    all_classes : numpy.ndarray
        All possible classes (superset of `classes_`).

    Returns
    -------
    numpy.ndarray
        Array of probabilities, ordered according to all_classes.

    """
    proba_ordered = np.zeros((probs.shape[0], all_classes.size), dtype=float)
    sorter = np.argsort(all_classes)  # http://stackoverflow.com/a/32191125/395857
    idx = sorter[np.searchsorted(all_classes, classes_, sorter=sorter)]
    proba_ordered[:, idx] = probs
    return proba_ordered


def url_is_reachable(url: str) -> bool:
    """Check if a URL is reachable.

    Parameters
    ----------
    url : str
        The URL.

    Returns
    -------
    bool
        Whether the URL is reachable.

    """
    try:
        r = requests.head(url, timeout=2)
        # successful responses and redirection messages
        # https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#information_responses
        return 100 <= r.status_code < 400
    except Exception:
        return False


def load_submission_log() -> pd.DataFrame:
    """Load the submission log.

    Returns
    -------
    df_sub_log : pandas.DataFrame
        The submission log,
        sorted by challenge score in descending order.

    """
    path = Path(__file__).parents[1] / "submissions"
    df_sub_log = pd.DataFrame.from_dict(yaml.safe_load(path.read_text())["Official Phase"], orient="index").sort_values(
        "score", ascending=False
    )
    return df_sub_log


def get_record_list_recursive3(
    db_dir: Union[str, bytes, os.PathLike],
    rec_patterns: Union[str, Dict[str, str]],
    relative: bool = True,
    with_suffix: bool = False,
) -> Union[List[str], Dict[str, List[str]]]:
    """Get the list of records in a recursive manner.

    For example, there are two folders "patient1", "patient2" in `db_dir`,
    and there are records "A0001", "A0002", ... in "patient1";
    "B0001", "B0002", ... in "patient2",
    then the output would be "patient1{sep}A0001", ..., "patient2{sep}B0001", ...,
    sep is determined by the system.

    Parameters
    ----------
    db_dir : `path-like`
        The parent (root) path of to search for records.
    rec_patterns : str or dict
        Pattern of the record filenames, e.g. ``"A(?:\\d+).mat"``,
        or patterns of several subsets, e.g. ``{"A": "A(?:\\d+).mat"}``
    relative : bool, default True
        Whether to return the relative path of the records.
    with_suffix : bool, default False
        Whether to include the suffix of the records.

    Returns
    -------
    List[str] or dict
        The list of records, in lexicographical order.

    """
    if isinstance(rec_patterns, str):
        res = []
    elif isinstance(rec_patterns, dict):
        res = {k: [] for k in rec_patterns.keys()}
    _db_dir = Path(db_dir).resolve()  # make absolute
    roots = [_db_dir]
    while len(roots) > 0:
        new_roots = []
        for r in roots:
            tmp = os.listdir(r)
            if isinstance(rec_patterns, str):
                res += [r / item for item in filter(re.compile(rec_patterns).search, tmp)]
            elif isinstance(rec_patterns, dict):
                for k in rec_patterns.keys():
                    res[k] += [r / item for item in filter(re.compile(rec_patterns[k]).search, tmp)]
            new_roots += [r / item for item in tmp if (r / item).is_dir()]
        roots = deepcopy(new_roots)
    if isinstance(rec_patterns, str):
        if with_suffix:
            res = [str((item.relative_to(_db_dir) if relative else item)) for item in res]
        else:
            res = [str((item.relative_to(_db_dir) if relative else item).with_suffix("")) for item in res]
        res = sorted(res)
    elif isinstance(rec_patterns, dict):
        for k in rec_patterns.keys():
            if with_suffix:
                res[k] = [str((item.relative_to(_db_dir) if relative else item)) for item in res[k]]
            else:
                res[k] = [str((item.relative_to(_db_dir) if relative else item).with_suffix("")) for item in res[k]]
            res[k] = sorted(res[k])
    return res
