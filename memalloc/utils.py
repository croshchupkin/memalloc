from wrapt import ObjectProxy


class MemoryBlockProxy(ObjectProxy):
    """
    A proxy class for a memoryview object, which allows us to mark it as a
    memory block which is allocated/free, and store the memoryview's offset
    relative to the start of the full buffer representing the available memory.
    """
    def __init__(self, wrapped: memoryview) -> None:
        super().__init__(wrapped)
        self._self_offset = None
        self._self_allocated = None

    @property
    def offset(self) -> int:
        return self._self_offset

    @offset.setter
    def offset(self, val: int) -> None:
        self._self_offset = val

    @property
    def allocated(self) -> bool:
        return self._self_allocated

    @allocated.setter
    def allocated(self, val: bool) -> None:
        self._self_allocated = val
