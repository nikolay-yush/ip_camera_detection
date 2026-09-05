import os
import sys

def silence_c_libraries_safe():
    """Safely redirects C-level stderr (descriptor 2) to devnull 
    without blocking execution or crashing on non-standard I/O.
    """
    try:
        STDERR_FD = 2 
        
        flags = os.O_WRONLY
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK

        devnull_fd = os.open(os.devnull, flags)

        os.dup2(devnull_fd, STDERR_FD)
        os.close(devnull_fd)

    except Exception:
        pass