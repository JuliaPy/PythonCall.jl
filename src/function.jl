"""
    pymethod(o)

Convert `o` to a Python instance method.
"""
pymethod(o) = check(C.PyInstanceMethod_New(pyobject(o)))
export pymethod

"""
    pyclassmethod(o)

Convert `o` to a Python class method.
"""
pyclassmethod(o) = pyclassmethodtype(o)
export pyclassmethod

"""
    pystaticmethod(o)

Convert `o` to a Python static method.
"""
pystaticmethod(o) = pystaticmethodtype(o)
export pystaticmethod

"""
    pyproperty(fget=pynone, fset=pynone, fdel=pynone, doc=pynone)

Create a Python property with the given getter, setter, deleter and docstring.
"""
pyproperty(args...; opts...) = pypropertytype(args...; opts...)
export pyproperty
