"""
Just some progress loging staff used when verbose is set to 1 in the solver
"""
from __future__ import annotations

import sys
import time


class ProgressLogger:
    """Hierarchical \r-based progress display for nested pipeline levels."""

    def __init__(self, D: int, verbose: int):
        self.verbose = verbose
        self.D = D
        self.level = -1
        self._last_len = 0  # length of the last thing written on the current line
        self._n_width = 0  # width to right-justify N, fixed from the root level's N

    def enter_level(self, N: int, D: int, N_ITER: int) -> _LevelCtx:
        """Push a new recursion level and return its context."""
        self.D = D
        self.level += 1
        if self.level == 0:
            self._n_width = len(str(N))  # root has the largest N; sets the alignment for all sub-levels
        return _LevelCtx(self, N, D, N_ITER, self.level)

    def exit_level(self) -> None:
        self.level -= 1

    def _prefix(self, level: int, N: int) -> str:
        return f"[Level {level} : N = {N:>{self._n_width}}] "

    def write(self, level: int, N: int, msg: str, newline: bool = False) -> None:
        if self.verbose < 1:
            return
        line = f"{self._prefix(level, N)}{msg}"
        # pad with spaces to fully overwrite whatever was on the line before
        pad = max(0, self._last_len - len(line))
        sys.stdout.write(f"\r{line}{' ' * pad}")
        if newline:
            sys.stdout.write("\n")
            self._last_len = 0
        else:
            self._last_len = len(line)
        sys.stdout.flush()


class _LevelCtx:
    """Tracks timing and tick state for a single pipeline level."""

    def __init__(self, logger: ProgressLogger, N: int, D: int, N_ITER: int, level: int):
        self._log = logger
        self.N = N
        self.D = D
        self.N_ITER = N_ITER
        self.level = level
        self._tick = 0
        self._t0: float | None = None
        self._t_iter: float | None = None
        self._t_start: float | None = None  # set by start(), once this level's own work begins

    def start(self) -> None:
        """Mark the start of this level's own work (after any child recursion)."""
        self._t_start = time.perf_counter()

    def _write(self, msg: str, newline: bool = False) -> None:
        self._log.write(self.level, self.N, msg, newline=newline)

    def on_bruteforce_start(self, eta_seconds: float | None = None) -> None:
        if eta_seconds is None:
            self._write("bruteforce…")
        else:
            self._write(f"bruteforce — ETA ~{self._fmt_duration(eta_seconds)}")

    def on_bruteforce_done(self) -> None:
        elapsed = time.perf_counter() - self._t_start
        self._write(f"done {self._fmt_duration(elapsed)}", newline=True)

    def tick(self) -> None:
        """Called once per gridification callback (= one full iteration). Drives the ETA display."""
        now = time.perf_counter()
        self._tick += 1
        if self._tick == 1:
            self._t0 = now
            self._write(f"{self._bar()} calibrating…")
            return
        if self._tick == 2:
            self._t_iter = now - self._t0  # type: ignore[operator]
        self._write(f"{self._bar()} — {self._eta(now)} remaining")

    def done(self) -> None:
        elapsed = time.perf_counter() - self._t_start
        self._write(f"done {self._fmt_duration(elapsed)}", newline=True)

    def _bar(self) -> str:
        filled = max(0, self._tick - 1)
        W = 20
        n_fill = int(W * filled / self.N_ITER)
        bar = "▓" * n_fill + "░" * (W - n_fill)
        return f"{filled}/{self.N_ITER} [{bar}]"

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds = round(seconds)
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes, rest = divmod(seconds, 60)
        return f"{minutes:.0f}min{rest:02.0f}"

    def _eta(self, now: float) -> str:
        if self._t_iter is None:
            return "?"
        remaining = (self.N_ITER - (self._tick - 1)) * self._t_iter
        return f"~{self._fmt_duration(remaining)}"