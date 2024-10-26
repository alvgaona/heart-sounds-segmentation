import os
from typing import Generator, Tuple, Union


def walk_files(
    root: str,
    suffix: Union[str, Tuple[str]],
    prefix: bool = False,
    remove_suffix: bool = False,
) -> Generator[str, None, None]:
    """List recursively all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or
        ('.jpg', '.png'). It uses the Python "str.endswith" method and
        is passed directly.
        prefix (bool, optional): If true, prepends the full path to each
        result, otherwise only returns the name of the files
        found (Default: ``False``).
        remove_suffix (bool, optional): If true, removes the suffix to
        each result defined in suffix, otherwise will return the result
        as found (Default: ``False``).
    """

    root = os.path.expanduser(root)

    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        # `dirs` is the list used in os.walk function and by sorting it
        # in-place here, we change the
        # behavior of os.walk to traverse sub directory alphabetically
        # see also
        # https://stackoverflow.com/questions/6670029/can-i-force-python3s-os-walk-to-visit-directories-in-alphabetical-order-how#comment71993866_6670926
        files.sort()
        for f in files:
            if f.endswith(suffix):
                if remove_suffix:
                    f = f[: -len(suffix)]

                if prefix:
                    f = os.path.join(dirpath, f)

                yield f
