import importlib.resources as pkg_resources


def __get_path(key):
    if key == "insolation":
        with pkg_resources.path(__package__, "Insolation.txt") as path:
            return path
    elif key == "retreat":
        with pkg_resources.path(__package__, "R_lookuptable.txt") as path:
            return path
    elif key == "tmp":
        with pkg_resources.path(__package__, "TMP_xz.txt") as path:
            return path
    else:
        raise KeyError(key)  # pragma: no cover


DEFAULT_DATAPATH_DICT = {
    key: __get_path(key) for key in ["insolation", "retreat", "tmp"]
}
"""A dictionary that matches data parts to paths to the data files."""
