"""Make a detached Modal run readable while it is still running.

A detached job's output otherwise lives only in `modal app logs`, which can fail with "Logs query hit
resource limit", and a job that hits its timeout returns nothing. `streaming(...)` tees stdout and
stderr to a file on the results volume and mirrors the run's artifact tree onto it on the same timer,
so progress can be read mid-flight and a crash or timeout leaves the traceback and every completed
cycle behind. Copying the tree picks up every artifact the loop writes without a per-writer hook.
The commit runs on a daemon thread because the work itself never yields.
"""
import pathlib
import shutil
import sys
import threading
import time
import traceback
from contextlib import contextmanager

COMMIT_EVERY_S = 60      # a volume commit is not free; a minute is well under the per-epoch cadence


class _Tee:
    """Stamps each line with elapsed seconds, in the file only, so per-epoch wall time is recoverable
    and a stalled run can be told from a slow one."""

    def __init__(self, console, handle):
        self._console, self._handle = console, handle
        self._t0 = time.monotonic()
        self._fresh = True

    def write(self, text: str) -> int:
        self._console.write(text)
        for piece in text.splitlines(keepends=True):
            if self._fresh and piece.strip():
                self._handle.write(f"[t+{time.monotonic() - self._t0:8.1f}s] ")
            self._handle.write(piece)
            self._fresh = piece.endswith("\n")
        return len(text)

    def flush(self) -> None:
        self._console.flush()
        self._handle.flush()


@contextmanager
def streaming(volume, path: pathlib.Path, mirror: tuple[pathlib.Path, pathlib.Path] | None = None,
              commit_every_s: float = COMMIT_EVERY_S):
    """Tee stdout+stderr to `path`, mirror `mirror=(src_tree, dst_tree)`, and commit `volume` every
    `commit_every_s`. `src_tree` may not exist yet — it is skipped until it does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", buffering=1)      # line-buffered: a killed container still leaves lines
    stop = threading.Event()

    def publish():
        handle.flush()
        if mirror is not None and mirror[0].exists():
            # copytree raises at the end of the walk if a file vanished mid-write; letting that escape
            # would kill the commit thread for good. The next pass recopies.
            try:
                shutil.copytree(mirror[0], mirror[1], dirs_exist_ok=True)
            except shutil.Error as e:
                print(f"[live_log] mirror incomplete this pass: {e}", flush=True)
        volume.commit()

    def commit_loop():
        while not stop.wait(commit_every_s):
            publish()

    pump = threading.Thread(target=commit_loop, daemon=True)
    pump.start()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(real_out, handle)
    sys.stderr = _Tee(real_err, handle)
    try:
        yield
    except BaseException:
        # onto the tee, before stdout is restored, so the traceback reaches the volume log too
        traceback.print_exc()
        raise
    finally:
        stop.set()
        pump.join(timeout=commit_every_s)      # no publish() racing the final one
        sys.stdout, sys.stderr = real_out, real_err
        publish()
        handle.close()
