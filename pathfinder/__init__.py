#!/usr/bin/env python3
"""
#####################################
# PATH FINDER MODULE                #
# Author J.Yellen                   #
#####################################
"""

from .matrix_handler import BinaryAcceptance
from .dfs import HDFS, WHDFS
from .result import Results

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"
