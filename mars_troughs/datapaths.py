from pathlib import Path


class _DataPaths:
    """
    A class for holding paths to data files.
    Do not reference directly. Use the global
    DATAPATHS variable instead.
    """

    DATA: Path = (Path(__file__) / ".." / ".." / "data").resolve()
    INSOLATION: Path = DATA / "Insolation.txt"
    RETREAT: Path = DATA / "R_lookuptable.txt"
    TMP: Path = DATA / "TMP_xz.txt"


DATAPATHS = _DataPaths()
"""Global object that holds paths."""
