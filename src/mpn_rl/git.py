import json
import subprocess
from pathlib import Path
from typing import TypedDict

# A sweep freezes its submission revision into a REVISION file next to this
# module inside the code snapshot, so jobs record the snapshot they actually ran
# rather than the live repo HEAD (which may have moved on by the time a job starts).
REVISION_FILE = Path(__file__).with_name("REVISION")


class Revision(TypedDict):
    commit: str
    dirty: bool


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def status() -> str:
    return _git("status", "--porcelain")


def is_dirty() -> bool:
    return bool(status())


def git_revision() -> Revision:
    if REVISION_FILE.exists():
        revision: Revision = json.loads(REVISION_FILE.read_text())
        return revision
    return {"commit": _git("rev-parse", "HEAD"), "dirty": is_dirty()}
