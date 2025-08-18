# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.__init__
# Brief : Package root; exposes subpackages and resolves version metadata.
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
spintransport package root and namespace helpers.
"""

from importlib.metadata import version, PackageNotFoundError

__all__ = ["io", "analysis", "viz", "reports", "physics", "cli"]

try:
    __version__ = version("spintransport")
except PackageNotFoundError:
    # Fallback for editable installs or when not built yet
    __version__ = "0.1.1"