"""
The module containing the implementation for the MemoryManager.
The allocation-specific documentation is located on the MemoryManager class.

The first-fit memory allocation with faux "memory-address"-ordered memory
blocks was chosen as a reasonable simple solution (basically a greatly simplified
version of g++ memory allocator, which is a first-fit roving pointer algorithm
enhanced with bins of different sizes). Further improvement of the memory management
approach will require further reading of the memory management-related
research papers, as the efficacy of algorithms and policies is not always evident
without thorough testing in conditions close to the real-world ones.
"""
from typing import Optional

from .exceptions import OutOfMemoryError
from .utils import MemoryBlockProxy


class MemoryManager:
    """
    A memory manager implementing the first-fit memory allocation. Large blocks
    will be divided into smaller ones, so that the allocated block is a perfect
    fit for the data.

    The buffer is required to be a bytearray instance, so that memoryview can
    be used to modify its data directly due to bytearray's support of the
    buffer protocol. AssertionError will be raised if it is not so.

    Block list is "address"-ordered, where the "address" is simply the
    index in the list of all blocks, and the aforementioned list is made to
    represent contiguous slices of memory.
    For simplicity, when a block is divided into smaller ones, it will just be
    removed from the list, and the resulting blocks will be inserted into
    its place in the correct order.

    For simplicity, this memory manager immediately tries to coalesce the
    freshly-freed block with the neighbouring free blocks. Note that this only
    happens on explicit `free` call, and if the memory were pre-split into the
    same-sized blocks during the memory manager creation, and if we tried to allocate
    a block larger than the initial block size, we would get an OutOfMemoryError.

    The pros of the chosen approach are the lack of internal fragmentation due to
    perfect-fit memory block allocation, as well as (possibly) better locality of
    reference when working with same-size memory blocks due to the search always
    starting at the beginning of the blocks' list. When working with same-size
    blocks, all of the "gaps" left after free will always be able to contain the
    newly-allocated block. It needs to be mentioned that in case of frequent frees
    of adjacent blocks these will be coalesced into a single large block, which
    will then lead to re-splitting of the said block when using `alloc` to allocate
    a new one. In cases of working with same-size blocks only, this may be
    optimized by allowing to turn off coalescence completely.

    Searching for a fitting block will take O(N) at worst, and *may* be improved
    by using a roving pointer to the block list and using a set of "bins"
    containing pre-split blocks of different sizes (similar to how g++ memory
    allocator works) - but this requires further reading of the reasearch papers
    on my side.

    Due to time constraints, all of the memory blocks are represented by
    memoryview objects into the buffer, created by the slicing of the main
    memoryview wrapping the buffer. The blocks are marked as free or allocated
    by assigning to the `allocated` boolean attribute dynamically added to each
    memoryview object, as well as the `offset` attribute indicating the
    offset in the buffer at which the block starts. I consider this solution as
    minimally viable, as memoryviews don't copy the buffer they are providing
    access to and allow the users to read and  mutate the part of the buffer
    exposed by the memoryview object as they see fit, as if it were a list.
    There are no pointers in python, so this is the only quick way to provide
    direct access to a bytearray's memory.

    The implementation is thread-safe due to the existence of GIL.
    If implemented in another language in a similar way, it will be prone to
    data races, and will require the use of synchronization constructs when
    accessing the buffer's contents and working with the list of blocks and its
    contents.

    At the moment most of the preconditions are simply asserted in the methods
    to speed up the development process.
    """
    def __init__(self,
                 buf: bytearray,
                 initial_block_size: Optional[int] = None) -> None:
        """
        Creates a memoryview into the provided buffer, to allow the
        interaction with sub-buffers such as slicing without creating new
        bytearray objects.
        All allocation and free operations will interact with the memoryview
        objects representing the memory blocks, as well as the main memoryview.
        As a side effect, this will also disallow buffer resizing until all
        of the memoryviews are released.

        `initial_block_size` is the size of initially available memory blocks
        in bytes. If None, there's initially a single block the size of the
        available memory.
        """
        assert isinstance(buf, bytearray), '`buf` must be a `bytearray`'

        self.buf_view = memoryview(buf)

        if initial_block_size is None:
            initial_block_size = len(buf)
        else:
            assert 0 < initial_block_size <= len(self.buf_view), '`initial_block_size` must be a positive int not greater than the buffer length'

        indexed_blocks = [
            (i, MemoryBlockProxy(self.buf_view[i:i+initial_block_size]))
            for i in range(0, len(self.buf_view), initial_block_size)]
        # mark all initial blocks as free and store the offset
        for idx, b in indexed_blocks:
            b.allocated = False
            b.offset = idx

        self.blocks = [b for (i, b) in indexed_blocks]

    def release(self) -> None:
        """
        Explicitly releases the memoryview for the full initial buffer,
        as well as all of the memoryviews representing the memory blocks.
        After this, alloc/free calls will always throw exceptions at the time
        attempts are made to work with the memoryviews.
        """
        self.buf_view.release()
        for b in self.blocks:
            b.release()

    def alloc(self, size: int) -> memoryview:
        """
        Allocates a part of the buffer.
        Returns a memoryview representing the allocated block.

        Walks the list of free blocks from the beginning.
        When it finds a large enough block, if it's larger than the requested
        amount of bytes, the block is divided into 2 blocks, with the first one
        being a perfect fit.
        If no free blocks of fitting size are available, raises an
        OutOfMemoryError.

        `size` is asserted to be a positive integer not greater than the size
        of the buffer.
        """
        assert 0 < size <= len(self.buf_view), '`size` must be a positive int not greater than the buffer length'

        for idx, block in enumerate(self.blocks):
            if size <= len(block) and not block.allocated:
                return self._alloc(idx, size)
        else:
            raise OutOfMemoryError(
                f'No memory blocks were available to allocate {size} bytes of memory')

    def free(self, block: memoryview):
        """
        Marks the block represented by a memoryview object as "free".
        Expects the memoryview to be the one returned by `alloc` by checking
        for the existence of an `allocated` boolean attribute and will throw
        an AssertionError if the attribute doesn't exist or is not set to True.

        NOTE: the memoryview describing the block is not `release()d`, so it is
        possible to create a "dangling pointer"-like situation where it is
        accessed by the outside code after free - and even after the block has
        been coalesced with another one. So the usage of blocks after free
        should be considered undefined behaviour.
        """
        assert isinstance(block, memoryview), '`block must be a memoryview`'
        assert hasattr(block, 'allocated'), '`block` must have an `allocated` attribute'
        assert block.allocated is True, '`block` must be an already allocated one'

        block_idx = self._find_block(block)
        block.allocated = False
        self._maybe_coalesce(block_idx)

    def _find_block(self, block: memoryview) -> int:
        """
        Returns the index of the block in the list of all blocks.
        Can most likely be improved by using binary search, as the list of
        blocks remains virtually sorted by their starting index relative to
        the main memoryview.
        """
        return self.blocks.index(block)

    def _alloc(self, idx: int, size: int) -> memoryview:
        """
        Allocates the block pointed to by the `idx`, potentially splitting
        it into two if the `size` is less than the size of the block
        """
        block = self.blocks[idx]
        block_size = len(block)
        if block_size > size:
            left = MemoryBlockProxy(block[:size])
            right = MemoryBlockProxy(block[size:])

            left.offset = block.offset
            right.offset = left.offset + len(left)
            right.allocated = False

            block.release()
            self.blocks.pop(idx)
            self.blocks.insert(idx, left)
            self.blocks.insert(idx + 1, right)

            allocated_block = left
        else:
            allocated_block = block

        allocated_block.allocated = True
        return allocated_block

    def _maybe_coalesce(self, block_idx: int) -> None:
        """
        If there are neighbouring free blocks, coalesce them with the block
        at `block_idx`.

        block_idx is asserted to be a correct index for the blocks' list.
        """
        assert 0 <= block_idx < len(self.blocks), '`block_idx` must be a correct block index'

        # how many old blocks to pop off the list
        num_pops = 1
        # the index in the block list to start removing the blocks at
        start_idx = block_idx

        left = None
        if block_idx - 1 >= 0:
            left = self.blocks[block_idx - 1]

        try:
            right = self.blocks[block_idx + 1]
        except IndexError:
            right = None

        block = self.blocks[block_idx]
        new_offset = block.offset
        new_len = len(block)

        if left is not None and not left.allocated:
            new_offset = left.offset
            new_len += len(left)
            start_idx = block_idx - 1
            num_pops += 1

        if right is not None and not right.allocated:
            new_len += len(right)
            num_pops += 1

        # If the new blok length if greater than the current one - that means
        # that we need to coalesce the blocks.
        if new_len > len(block):
            coalesced_block = MemoryBlockProxy(
                self.buf_view[new_offset:new_offset + new_len])
            coalesced_block.allocated = False
            coalesced_block.offset = new_offset

            for _ in range(num_pops):
                b = self.blocks.pop(start_idx)
                b.release()

            self.blocks.insert(start_idx, coalesced_block)
