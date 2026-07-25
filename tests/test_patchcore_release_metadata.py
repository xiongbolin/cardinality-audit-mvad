from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_patchcore_release_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "scripts" / "verify_patchcore_release_metadata.py")],
        cwd=root,
        check=True,
    )
