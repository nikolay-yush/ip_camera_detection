import os
import sys


def silence_c_libraries():
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stderr.fileno())
        os.close(devnull)
    except Exception:
        pass

