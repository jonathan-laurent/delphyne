import sys
from typing import Any


class StatusIndicator:
    def __init__(self, file: Any = sys.stdout, show: bool = False):
        self.last_len = [0]
        self.file = file
        self.show = show

    def on_status(self, msg: str) -> None:
        if not self.show:
            return
        pad = max(0, self.last_len[0] - len(msg))
        print(f"\r{msg}{' ' * pad}", end="", file=self.file, flush=True)
        self.last_len[0] = len(msg)

    def done(self) -> None:
        if self.show and self.last_len[0] > 0:
            print(f"\r{' ' * self.last_len[0]}\r", end="", file=self.file)
