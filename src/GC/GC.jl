"""
    module PythonCall.GC

Garbage collection of Python objects.

See `disable` and `enable`.
"""
module GC

"""
    PythonCall.GC.disable()

Do nothing.

!!! note

    In earlier versions of PythonCall, this function disabled the PythonCall garbage
    collector. This is no longer required because Python objects now have thread-safe
    finalizers. This function will be removed in PythonCall v1.
"""
disable() = nothing

"""
    PythonCall.GC.enable()

Do nothing.

!!! note

    In earlier versions of PythonCall, this function re-enabled the PythonCall garbage
    collector. This is no longer required because Python objects now have thread-safe
    finalizers. This function will be removed in PythonCall v1.
"""
enable() = nothing

end # module GC
