aspyfunc(o) = o isa PyObject ? o : pyjlfunction(o)
aspyfuncornone(o) = o === nothing ? pynone : o isa PyObject ? o : pyjlfunction(o)

"""
    pymethod(o)

Convert `o` to a Python instance method.
"""
pymethod(o) = check(C.PyInstanceMethod_New(aspyfunc(o)))
export pymethod

"""
    pyclassmethod(o)

Convert `o` to a Python class method.
"""
pyclassmethod(o) = pyclassmethodtype(aspyfunc(o))
export pyclassmethod

"""
    pystaticmethod(o)

Convert `o` to a Python static method.
"""
pystaticmethod(o) = pystaticmethodtype(aspyfunc(o))
export pystaticmethod

"""
    pyproperty(fget=pynone, fset=pynone, fdel=pynone, doc=pynone)

Create a Python property with the given getter, setter, deleter and docstring.
"""
pyproperty(_fget=pynone, _fset=pynone, _fdel=pynone, _doc=pynone; fget=_fget, fset=_fset, fdel=_fdel, doc=_doc) =
    pypropertytype(aspyfuncornone(fget), aspyfuncornone(fset), aspyfuncornone(fdel), pyobject(doc))
export pyproperty
