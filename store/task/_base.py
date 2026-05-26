"""store/memory/_base.py — 异步 Store 基类。"""
from __future__ import annotations

from typing import Callable

import aiosqlite


class BaseAsyncStore:
    """所有异步 Store 子类的公共基础。

    子类只需传入 db_getter 即可，不必重复实现 __init__ 和 _db 属性。
    """

    def __init__(self, db_getter: Callable[[], aiosqlite.Connection]) -> None:
        self._db_getter = db_getter

    @property
    def _db(self) -> aiosqlite.Connection:
        return self._db_getter()
