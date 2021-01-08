from unittest import TestCase

from memalloc.exceptions import OutOfMemoryError
from memalloc.memory_manager import MemoryManager


class MemoryManagerTestCommonMixin:
    def test_fragmentation_mitigated_direct_pass(self):
        chars = [self.mm.alloc(1) for _ in range(len(self.buf))]

        for idx, item in enumerate(chars):
            self.mm.free(item)
        self.mm.alloc(len(self.buf))

        self.assertEqual(len(self.mm.blocks), 1)
        self.assertTrue(self.mm.blocks[0].allocated)

    def test_fragmentation_mitigated_reverse_pass(self):
        chars = [self.mm.alloc(1) for _ in range(len(self.buf))]

        for idx, item in enumerate(reversed(chars)):
            self.mm.free(item)
        self.mm.alloc(len(self.buf))

        self.assertEqual(len(self.mm.blocks), 1)
        self.assertTrue(self.mm.blocks[0].allocated)

    def test_fragmentation_mitigated_2_step_free(self):
        chars = [self.mm.alloc(1) for _ in range(len(self.buf))]

        for idx, item in enumerate(chars[::2]):
            self.mm.free(item)
        for idx, item in enumerate(chars[1::2]):
            self.mm.free(item)
        self.mm.alloc(len(self.buf))

        self.assertEqual(len(self.mm.blocks), 1)
        self.assertTrue(self.mm.blocks[0].allocated)

    def test_fragmentation_mitigated_2_step_free_reversed(self):
        chars = [self.mm.alloc(1) for _ in range(len(self.buf))]

        for idx, item in enumerate(reversed(chars[::2])):
            self.mm.free(item)

        for idx, item in enumerate(reversed(chars[1::2])):
            self.mm.free(item)
        self.mm.alloc(len(self.buf))

        self.assertEqual(len(self.mm.blocks), 1)
        self.assertTrue(self.mm.blocks[0].allocated)


class TestMemoryManager(TestCase, MemoryManagerTestCommonMixin):
    def setUp(self):
        # initialize buffer with a range of integers for convenience
        self.buf = bytearray([i for i in range(10)])
        self.mm = MemoryManager(self.buf)

    def test_alloc(self):
        self.mm.alloc(3)
        self.assertEqual(len(self.mm.blocks), 2)
        self.assertEqual(
            [(len(b), b.allocated, b.offset, list(b)) for b in self.mm.blocks],
            [(3, True, 0, list(self.buf[:3])),
             (7, False, 3, list(self.buf[3:]))])

    def test_buffer_values_mutated(self):
        block = self.mm.alloc(5)
        self.assertEqual(len(block), 5)
        new_values = [i for i in range(251, 256)]
        for idx, val in enumerate(new_values):
            block[idx] = val

        self.assertEqual(list(self.buf[:5]), new_values)


class TestMemoryManagerPresplitBlocks(TestCase, MemoryManagerTestCommonMixin):
    """
    Same as TestMemoryManager - but the memory is pre-split into 1-byte blocks
    """
    def setUp(self):
        self.buf = bytearray([i for i in range(10)])
        self.mm = MemoryManager(self.buf, 1)

    def test_alloc_throws_out_of_memory(self):
        with self.assertRaises(OutOfMemoryError):
            self.mm.alloc(3)
