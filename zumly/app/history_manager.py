"""Small transactional helpers for continuous editor gestures."""

from __future__ import annotations

import copy
from typing import Generic, TypeVar


T = TypeVar("T")


class DragTransaction(Generic[T]):
    """Hold a drag's before/current values until mouse release.

    The transaction deliberately has no knowledge of Qt or the undo stack.
    The owner decides how to persist the single committed value, while all
    intermediate updates remain ephemeral.
    """

    def __init__(self) -> None:
        self._before: T | None = None
        self._current: T | None = None

    @property
    def active(self) -> bool:
        return self._before is not None

    @property
    def before(self) -> T | None:
        return copy.deepcopy(self._before)

    def begin(self, value: T) -> None:
        self._before = copy.deepcopy(value)
        self._current = copy.deepcopy(value)

    def update(self, value: T) -> None:
        if not self.active:
            return
        self._current = copy.deepcopy(value)

    def finish(self) -> T | None:
        if not self.active:
            return None
        before = self._before
        current = copy.deepcopy(self._current)
        self._before = None
        self._current = None
        if current == before:
            return None
        return current

    def cancel(self) -> T | None:
        if not self.active:
            return None
        before = copy.deepcopy(self._before)
        self._before = None
        self._current = None
        return before
