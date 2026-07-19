"""Run a verifier and bypass unstable native-library process teardown on Windows."""

from __future__ import annotations

import os
import runpy
import sys
import traceback
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_verifier_safely.py SCRIPT [ARG ...]")
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    exit_code = 0
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code is None:
            exit_code = 0
        elif isinstance(exc.code, int):
            exit_code = exc.code
        else:
            print(exc.code, file=sys.stderr)
            exit_code = 1
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
