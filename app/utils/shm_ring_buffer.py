import numpy as np
from multiprocessing import shared_memory


class SharedMemoryRingBuffer:
    """Manages a ring buffer in shared memory for zero-copy numpy array transfers across processes."""

    def __init__(
        self,
        name_prefix: str,
        num_slots: int = 4,
        shape: tuple = (1080, 1920, 3),
        dtype=np.uint8,
        create: bool = False,
    ):
        self.num_slots = num_slots
        self.shape = shape
        self.dtype = dtype
        self.frame_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

        self.shm_blocks: list[shared_memory.SharedMemory] = []
        self.arrays: list[np.ndarray] = []
        self.write_idx = 0

        for i in range(num_slots):
            shm_name = f"{name_prefix}_slot_{i}"
            if create:
                shm = shared_memory.SharedMemory(
                    name=shm_name, create=True, size=self.frame_bytes
                )
            else:
                shm = shared_memory.SharedMemory(name=shm_name, create=False)

            self.shm_blocks.append(shm)
            # Create a zero-copy NumPy array directly over the shared buffer memory
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            self.arrays.append(arr)

    def write_next(self, frame: np.ndarray) -> tuple[int, str]:
        """Copies frame bytes into the current slot and returns slot index & SHM name."""
        slot_idx = self.write_idx
        np.copyto(self.arrays[slot_idx], frame)
        shm_name = self.shm_blocks[slot_idx].name

        # Advance write pointer in ring topology
        self.write_idx = (self.write_idx + 1) % self.num_slots
        return slot_idx, shm_name

    def get_array(self, slot_idx: int) -> np.ndarray:
        """Returns direct zero-copy reference to the frame stored in specified slot."""
        return self.arrays[slot_idx]

    def close(self):
        """Closes memory mappings for the current process."""
        for shm in self.shm_blocks:
            try:
                shm.close()
            except Exception:
                pass

    def unlink(self):
        """Frees shared memory region from the OS kernel (Call only from master process)."""
        for shm in self.shm_blocks:
            try:
                shm.unlink()
            except Exception:
                pass