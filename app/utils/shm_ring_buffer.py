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
                # Force cleanup of orphaned shared memory segments from previous crashed runs
                try:
                    temp_shm = shared_memory.SharedMemory(name=shm_name, create=False)
                    temp_shm.close()
                    temp_shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass

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

        # Validate incoming frame dimensions and dtype to prevent crashes
        if frame.shape != self.shape or frame.dtype != self.dtype:
            raise ValueError(
                f"[SHM] Frame shape/dtype mismatch! Expected {self.shape} ({self.dtype}), got {frame.shape} ({frame.dtype})"
            )

        np.copyto(self.arrays[slot_idx], frame)
        shm_name = self.shm_blocks[slot_idx].name

        # Advance write pointer in ring topology
        self.write_idx = (self.write_idx + 1) % self.num_slots
        return slot_idx, shm_name

    def get_array(self, slot_idx: int) -> np.ndarray | None:
        """Returns direct zero-copy reference to the frame stored in specified slot."""
        if 0 <= slot_idx < self.num_slots and self.arrays:
            return self.arrays[slot_idx]
        return None

    def close(self) -> None:
        """Closes memory mappings for the current process safely releasing NumPy exports."""
        # CRITICAL: Clear NumPy buffer references first to allow underlying C-mmap to unbind
        self.arrays.clear()

        for shm in self.shm_blocks:
            try:
                shm.close()
            except Exception:
                pass
        self.shm_blocks.clear()

    def unlink(self) -> None:
        """Frees shared memory region from the OS kernel (Call only from master process)."""
        self.close()

        # Re-attach temporarily if needed or unlink directly if references exist
        # Note: shm.unlink() must be called once by the creating process
        for shm in self.shm_blocks:
            try:
                shm.unlink()
            except Exception:
                pass