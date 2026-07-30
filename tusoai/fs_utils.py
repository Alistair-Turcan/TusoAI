from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path
from typing import Any

_IGNORABLE_METADATA_ERRNOS = {
    errno.EPERM,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
}


def is_ignorable_metadata_error(exc: BaseException) -> bool:
    """Return True for filesystem metadata errors common on object-store FUSE mounts."""
    err_no = getattr(exc, "errno", None)
    return isinstance(exc, OSError) and err_no in _IGNORABLE_METADATA_ERRNOS


def copyfile_portable(
    src: str | Path,
    dst: str | Path,
    *,
    follow_symlinks: bool = True,
) -> str:
    """Copy file contents without applying metadata unsupported by S3-FUSE.

    This has the destination-directory behavior and return value of
    :func:`shutil.copy`, but deliberately uses :func:`shutil.copyfile` rather
    than applying the source mode with ``copymode``. The omitted mode update is
    the part of ``shutil.copy`` that commonly fails on object-store mounts.
    """
    destination = os.fspath(dst)
    if Path(destination).is_dir():
        destination = os.path.join(destination, os.path.basename(os.fspath(src)))
    shutil.copyfile(src, destination, follow_symlinks=follow_symlinks)
    return destination


def copytree_portable(src: str | Path, dst: str | Path, **kwargs: Any) -> Any:
    """Copy a tree without failing on chmod/utime metadata operations.

    S3-FUSE and similar object-store mounts can allow file data copies while
    rejecting metadata updates such as chmod or utime. ``shutil.copytree``
    performs ``copystat`` on directories even when its file copy function does
    not preserve metadata, so we temporarily make ``copystat`` swallow those
    mount-specific errors.
    """
    kwargs.setdefault("copy_function", copyfile_portable)
    original_copystat = shutil.copystat

    def _safe_copystat(source: str | Path, destination: str | Path, *args: Any, **inner_kwargs: Any) -> None:
        try:
            original_copystat(source, destination, *args, **inner_kwargs)
        except OSError as exc:
            if not is_ignorable_metadata_error(exc):
                raise

    shutil.copystat = _safe_copystat
    try:
        return shutil.copytree(src, dst, **kwargs)
    finally:
        shutil.copystat = original_copystat


def rmtree_portable(path: str | Path, **kwargs: Any) -> None:
    """Remove a tree while ignoring object-store FUSE rmdir metadata errors."""
    if kwargs.get("ignore_errors"):
        shutil.rmtree(path, **kwargs)
        return

    def _onerror(func: Any, failed_path: str, exc_info: tuple[type[BaseException], BaseException, Any]) -> None:
        exc = exc_info[1]
        if getattr(func, "__name__", "") == "rmdir" and is_ignorable_metadata_error(exc):
            return
        if is_ignorable_metadata_error(exc):
            return
        raise exc

    kwargs.setdefault("onerror", _onerror)
    shutil.rmtree(path, **kwargs)
