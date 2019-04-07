import re
import errno
import os


def tryfloat(s):
    try:
        return float(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryfloat(c) for c in re.split('([0-9.]+)', s)]


def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    return sorted(l, key=alphanum_key)


def get_center(x):
    return (x - 1.) / 2.


def mkdir_p(path):
    """mimic the behavior of mkdir -p in bash"""
    try:
        os.makedirs(path)
    except OSError as exc:    # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
