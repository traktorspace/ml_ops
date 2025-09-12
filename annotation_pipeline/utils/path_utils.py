import io
import os
import re
import shutil
import zipfile
from pathlib import Path

import fs.path
from fs.base import FS  # pyfilesystem2 (pip install fs)


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
) -> tuple[Path, Path]:
    """
    Retrieve the data-annotation files pair for a given product.

    Parameters
    ----------
    root_dir :
        Root directory that contains product sub-directories.
    prod_name :
        Name of the product (sub-directory) to look for.
    annotation_suffix :
        Suffix appended to the product level to locate the
        corresponding annotation file.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Paths to the cube (`*.tif`) and its annotation (`*_annotation.tif`)
        in the order ``(cube_path, annotation_path)``.

    Raises
    ------
    FileNotFoundError
        If the product directory does not exist.
    IndexError
        If the product level cannot be inferred.
    """
    dst_path = root_dir / prod_name
    if not dst_path.exists():
        raise FileNotFoundError(f'{prod_name} not found under {root_dir}')

    prod_level = infer_product_level(dst_path)
    if prod_level is None:
        raise IndexError(f"Can't infer product level from {dst_path}")

    cube_path = dst_path / f'{prod_level}.tif'
    annotation_path = dst_path / f'{prod_level}_{annotation_suffix}.tif'

    if not cube_path.exists() or not annotation_path.exists():
        raise FileNotFoundError(
            f"Missing cube or annotation file for product '{prod_name}'. "
            f'Expected: {cube_path}, {annotation_path}'
        )

    return cube_path, annotation_path


def unzip_any(
    src: str | Path | io.IOBase,
    dst: str | Path,
    *,
    src_fs: FS | None = None,
    dst_fs: FS | None = None,
) -> None:
    """
    Extract a ZIP archive from *any* location to *any* location
    (host-disk ↔ pyfilesystem2 FS) in a single call.

    The function transparently handles all four combinations:

    1. disk → disk
    2. disk → FS (e.g. ``MemoryFS``, S3, FTP, …)
    3. FS   → disk
    4. FS   → FS

    Parameters
    ----------
    src : str | pathlib.Path | io.IOBase
        The ZIP archive to read.

        * When ``src_fs`` is ``None`` and *src* is a string/Path, it is
          interpreted as a path on the host file-system.

        * When ``src_fs`` is an :class:`fs.base.FS` instance, *src* is
          interpreted as a path **inside** that filesystem.

        * When *src* is already an open, readable file-like object
          (``BytesIO``, socket, etc.) it is used directly.
    dst : str | pathlib.Path
        Destination directory.  As with *src*, its meaning depends on
        ``dst_fs``.
    src_fs : fs.base.FS, optional
        Filesystem that *contains* ``src``.  Leave ``None`` when *src*
        resides on disk or is a file-like object.
    dst_fs : fs.base.FS, optional
        Filesystem that will receive the extracted files.  Leave
        ``None`` to write to the host disk.

    Returns
    -------
    None
        All files/folders are created as a side-effect.

    Examples
    --------
    Disk → disk
    >>> unzip_any('/tmp/archive.zip', '/tmp/out')

    Disk → MemoryFS
    >>> from fs.memoryfs import MemoryFS
    >>> mem_fs = MemoryFS()
    >>> unzip_any('/tmp/archive.zip', '/unzipped', dst_fs=mem_fs)

    MemoryFS → disk
    >>> unzip_any('data.zip', '/tmp/out2', src_fs=mem_fs)

    MemoryFS → MemoryFS
    >>> dst_mem = MemoryFS()
    >>> unzip_any('data.zip', '/', src_fs=mem_fs, dst_fs=dst_mem)
    """
    # ---------- open the archive --------------------------------
    if isinstance(src, io.IOBase) and not isinstance(src, (str, Path)):
        zf = zipfile.ZipFile(src, 'r')
    elif src_fs is None:
        zf = zipfile.ZipFile(str(src), 'r')
    else:  # src inside a pyfilesystem FS
        zf = zipfile.ZipFile(src_fs.open(src, 'rb'), 'r')

    # ---------- ensure destination root exists ------------------
    if dst_fs is None:
        Path(dst).mkdir(parents=True, exist_ok=True)
    else:
        dst_fs.makedirs(dst, recreate=True)

    # ---------- extract each member -----------------------------
    for info in zf.infolist():
        rel = info.filename.rstrip('/')
        if not rel:  # ZIP’s top-level dir entry
            continue

        if dst_fs is None:  # writing to host disk
            target = Path(dst, rel)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src_f, open(target, 'wb') as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
        else:  # writing into any FS object
            target = fs.path.combine(dst, rel)
            if info.is_dir():
                dst_fs.makedirs(target, recreate=True)
            else:
                dst_fs.makedirs(fs.path.dirname(target), recreate=True)
                with zf.open(info) as src_f, dst_fs.open(target, 'wb') as dst_f:
                    shutil.copyfileobj(src_f, dst_f)

    zf.close()


def parse_s3_path(
    prodname: str, prod_path: Path, use_cloud_data_source: bool
) -> str:
    """
    Return the remote object key or local Geo-TIFF path for *prodname*.

    When ``use_cloud_data_source`` is ``True`` the function extracts the
    portion of *prod_path* that follows ``"/processed/"`` and appends
    ``".zip"``; this matches the naming convention in the cloud bucket.
    Otherwise it appends the expected Geo-TIFF name
    ``"<product_level>.tif"`` to *prod_path*.

    This function acts as a workaround to use batman server paths and find
    a match with google bucket paths.

    Parameters
    ----------
    prodname : str
        Product name (e.g. ``"hyperfield1a_L1C_20250310T142453"``).
    prod_path : pathlib.Path
        Full path to the product directory on disk.
    use_cloud_data_source : bool
        ``True`` to build the cloud-bucket key, ``False`` to build the
        on-disk Geo-TIFF path.

    Returns
    -------
    str
        • Cloud case: object key like
          ``"2025/03/10/hyperfield-1a/hyperfield1a_L1C_20250310T142453.zip"``
        • Local case: full path to the Geo-TIFF on disk.

    Raises
    ------
    ValueError
        If ``use_cloud_data_source`` is ``True`` and *prod_path* does not
        contain the segment ``"/processed/"``.

    Examples
    --------
    >>> parse_s3_path(
    ...     "hyperfield1a_L1C_20250310T142453",
    ...     Path("/bigdata/hyperfield/processed/2025/03/10/"
    ...          "hyperfield-1a/hyperfield1a_L1C_20250310T142453"),
    ...     True,
    ... )
    '2025/03/10/hyperfield-1a/hyperfield1a_L1C_20250310T142453.zip'

    >>> parse_s3_path("hyperfield1a_L1C_20250310T142453", Path("/tmp/hyperfield1a"), False)
    '/tmp/hyperfield1a/hyperfield1a_L1C_20250310T142453/L1C.tif'
    """
    if use_cloud_data_source:
        match = re.search(r'(?<=/processed/).+$', str(prod_path))
        if match is None:  # defensive: processed/ not found
            raise ValueError(f"'processed/' not found in {prod_path}")
        return f'{match.group(0)}.zip'

    # local (non-cloud) case
    return str(prod_path / f'{infer_product_level(prodname)}.tif')
