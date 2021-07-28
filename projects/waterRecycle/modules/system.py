"""system.py
Development of the functions and classes that deal with system calls
"""
import sys

def ensure_config():
    """
    Use assert statements to ensure the system callers are using the
    same base system configuration
    """
    local_version = sys.version_info
    assert local_version.major == 3, "Not the right major version of python"
    assert local_version.minor == 9, "Not the right minor version of python"
