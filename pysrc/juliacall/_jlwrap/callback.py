"""Helpers for wrapping Julia callbacks as Python functions."""


def wrap_callback(f):
    return lambda *args, **kwargs: f(*args, **kwargs)
