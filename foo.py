from memalloc.memory_manager import MemoryManager

a = bytearray([i for i in range(10)])
mm = MemoryManager(a)
foo_short = mm.alloc(2)
foo_short[1] = 255
foo_int = mm.alloc(4)
foo_int[2] = 255
mm.free(foo_int)
import ipdb; ipdb.set_trace()
