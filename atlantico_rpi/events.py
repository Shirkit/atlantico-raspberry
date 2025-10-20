"""Simple EventQueue for signaling work between threads."""

from dataclasses import dataclass
from queue import Queue, Empty
from typing import Any, Optional


@dataclass
class Event:
    name: str
    payload: Optional[Any] = None


class EventQueue:
    def __init__(self, maxsize: int = 0):
        self._q = Queue(maxsize=maxsize)

    def put(self, name: str, payload: Optional[Any] = None, block: bool = True, timeout: Optional[float] = None) -> None:
        self._q.put(Event(name, payload), block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Event:
        return self._q.get(block=block, timeout=timeout)

    def try_get(self) -> Optional[Event]:
        try:
            return self._q.get(block=False)
        except Empty:
            return None

    def task_done(self) -> None:
        self._q.task_done()

    def empty(self) -> bool:
        return self._q.empty()
