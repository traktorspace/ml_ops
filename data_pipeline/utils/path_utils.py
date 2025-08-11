import os
import re
from pathlib import Path


def get_prod_list_from_file(filepath: Path):
    """
    Return list of file paths given a txt file.

    Parameters
    ----------
    filepath
        path of the txt file containing the list of products
    """
    with open(filepath) as file:
        lines = file.readlines()
    prods = [
        line.rstrip('\n') for line in lines if os.path.isfile(line.rstrip('\n'))
    ]
    if len(prods) != len(lines):
        raise ValueError(
            f'Not all the products in the list exists. Expected: {len(lines)}, found: {len(prods)}'
        )
    return prods


def get_parent_from_path(path: Path):
    """Return parent folder given a Path"""
    return path.parent.stem


def get_parents_from_paths(paths: list[Path]):
    """Invoke get_parent_from_path for each element in a list of Paths"""
    return [get_parent_from_path(p) for p in paths]


def infer_product_level(product_path: Path | str):
    """
    Extract the product-level token (e.g., ``L1B``) from a filename.

    The filename is expected to follow the pattern
    ``<name>_<level>_<timestamp>``, where *level* starts with ``L``
    followed by one or more digits and a single uppercase letter.

    Parameters
    ----------
    product_path
        Path object pointing to the product file whose name encodes
        the product level.

    Returns
    -------
    str or None
        The extracted level (e.g., ``"L1B"``) if the pattern is found;
        otherwise ``None``.

    Examples
    --------
    >>> from pathlib import Path
    >>> infer_product_level(Path("hyperfield1a_L1B_20250507T101305"))
    'L1B'
    >>> infer_product_level(Path("invalid_name"))
    None
    """
    if isinstance(product_path, str):
        product_path = Path(product_path)
    match = re.search(r'_(?P<level>L\d+[A-Z])_', product_path.name)
    if match:
        return match['level']
    else:
        return None


def tif_exists(row) -> bool:
    """Given a row (tuple) from db return if a TIF file exist at that address.

    Parameters
    ----------
    row
        Tuple containing (id, product_id, root_dir)\n
        _Example_: (abc123, hyperfield1a_L1B_20250505T185424, /somepath/product_id)

    Returns
    -------
        True if the path exists, False otherwise.
    """
    _, product_id, root_dir = row
    return (
        Path(root_dir) / f'{infer_product_level(Path(product_id))}.tif'
    ).exists()


def make_symlink(
    path_to_link: str | os.PathLike, destination: str | os.PathLike
) -> Path:
    """
    Create a symbolic link ``destination`` → ``path_to_link``.

    Safety rules
    ------------
    1. ``path_to_link`` **must exist** (avoids creating a broken link).
    2. ``destination`` **must NOT exist**.  Any file / dir / symlink at that
       location raises ``FileExistsError`` so nothing is overwritten.

    Parameters
    ----------
    path_to_link :
        Existing file or directory that the link will point to.
    destination  :
        Filesystem location where the symlink will be created.

    Returns
    -------
    pathlib.Path
        Path object representing the newly-created symlink.
    """
    src = Path(path_to_link)
    link = Path(destination)

    # ---- Rule 1 -------------------------------------------------------------
    if not src.exists():
        raise FileNotFoundError(f'Source for symlink does not exist: {src}')

    # ---- Rule 2 -------------------------------------------------------------
    if link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            raise FileExistsError(
                'Safety error!\n'
                f'Destination already exists: {link}.\n'
                'You are risking overwriting the file.\n'
                'Please verify that you are using the correct function parameters.'
            )
    # ---- Create the link (prefer relative path) ----------------------------
    try:
        rel_src = os.path.relpath(src, link.parent)
        link.symlink_to(rel_src)
    except ValueError:  # e.g. different drive on Windows
        link.symlink_to(src)

    return link


def ensure_dir(dir_path: str | os.PathLike) -> Path:
    """
    Create *dir_path* (including parents) if it doesn't exist and return it as Path.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_data_pair(
    root_dir: Path, prod_name: str, annotation_suffix: str = 'annotation'
):
    dst_path = root_dir / prod_name
    if dst_path.exists():
        prod_level = infer_product_level(dst_path)
        cube_path = Path(f'{dst_path / prod_level}.tif')
        annotation_path = Path(
            f'{dst_path / prod_level}_{annotation_suffix}.tif'
        )

        if cube_path.exists() and annotation_path.exists():
            return cube_path, annotation_path
    else:
        raise FileNotFoundError(f'{prod_name} not found under {root_dir}')
