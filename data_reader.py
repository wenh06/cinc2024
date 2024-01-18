"""
"""

from typing import Any

from torch_ecg.databases.base import DataBaseInfo, PhysioNetDataBase
from torch_ecg.utils.misc import add_docstring

__all__ = [
    "CINC2024Reader",
]


_CINC2024_INFO = DataBaseInfo(
    title="""
    Digitization and Classification of ECG Images
    """,
    about="""
    """,
    usage=[
        "",
    ],
    note="""
    """,
    issues="""
    """,
    references=[
        "https://moody-challenge.physionet.org/2024/",
    ],
    # doi=["https://doi.org/10.13026/rjbz-cq89"],
)


@add_docstring(_CINC2024_INFO.format_database_docstring(), mode="prepend")
class CINC2024Reader(PhysioNetDataBase):
    """ """

    __name__ = "CINC2024Reader"

    def __init__(
        self,
        db_dir: str,
        verbose: int = 2,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in a dataframe.
        """
        raise NotImplementedError
