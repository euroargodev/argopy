"""
This submodule is a naive approach to massively load Argo data as a large collection of points, or profiles interpolated on standard pressure levels.

This is an experimental submodule based on previous work from the EA-RISE project.
This submodule is adapted from the original OSnet codebase developped by Sean Tokunaga and Guillaume Maze, but never published before.
https://github.com/euroargodev/argopy-osnet

API may change anytime, not all files may be necessary in here and shall be removed/refactored without notice.
"""
from argopy.utils.arco.fetcher import MassFetcher

__all__ = ('MassFetcher')