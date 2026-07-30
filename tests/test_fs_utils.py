import errno
import os
import shutil
from pathlib import Path

from tusoai.fs_utils import copyfile_portable, copytree_portable, rmtree_portable


def test_copyfile_portable_copies_data_without_mode_updates(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "source.txt"
    destination_dir = tmp_path / "destination"
    src.write_text("ok", encoding="utf-8")
    destination_dir.mkdir()

    def failing_copymode(*args, **kwargs):
        raise AssertionError("copyfile_portable must not update file modes")

    monkeypatch.setattr(shutil, "copymode", failing_copymode)

    copied_path = copyfile_portable(src, destination_dir)

    assert copied_path == str(destination_dir / src.name)
    assert Path(copied_path).read_text(encoding="utf-8") == "ok"


def test_copytree_portable_swallows_copystat_metadata_errors(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "data.txt").write_text("ok", encoding="utf-8")

    def failing_copystat(*args, **kwargs):
        raise OSError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(shutil, "copystat", failing_copystat)

    copytree_portable(src, dst)

    assert (dst / "data.txt").read_text(encoding="utf-8") == "ok"


def test_rmtree_portable_swallows_rmdir_metadata_errors(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()

    def fake_rmtree(path, **kwargs):
        onerror = kwargs["onerror"]
        onerror(os.rmdir, str(path), (OSError, OSError(errno.EPERM, "operation not permitted"), None))

    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)

    rmtree_portable(target)
