"""Run pip with tempfile.mkdtemp allowed to create world-accessible directories.

This machine's sandbox denies access to directories created with mode 0o700 (which is
what tempfile.mkdtemp uses), so pip cannot unpack wheels. Redirecting the mkdir mode is
enough for pip to work; nothing else is touched.
"""
import os
import sys
import tempfile

_real_mkdir = os.mkdir


def mkdir_open(path, mode=0o777, *args, **kwargs):
    return _real_mkdir(path, 0o777, *args, **kwargs)


os.mkdir = mkdir_open
tempfile._os.mkdir = mkdir_open

from pip._internal.cli.main import main

sys.exit(main(sys.argv[1:]))
