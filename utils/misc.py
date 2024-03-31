"""
Miscellaneous functions.
"""

from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests
import yaml

__all__ = [
    "func_indicator",
    "predict_proba_ordered",
    "url_is_reachable",
    "load_submission_log",
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
        return 100 <= r.status_code < 300
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
