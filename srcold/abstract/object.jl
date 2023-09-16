"""
    pyis(x, y)

True if `x` and `y` are the same Python object. Equivalent to `x is y` in Python.
"""
pyis(x, y) = @autopy x y getptr(x_) == getptr(y_)
export pyis

pyisnot(x, y) = !pyis(x, y)

"""
    pyrepr(x)

Equivalent to `repr(x)` in Python.
"""
pyrepr(x) = pynew(errcheck(@autopy x C.PyObject_Repr(getptr(x_))))
pyrepr(::Type{String}, x) = (s=pyrepr(x); ans=pystr_asstring(s); pydel!(s); ans)
export pyrepr

"""
    pyascii(x)

Equivalent to `ascii(x)` in Python.
"""
pyascii(x) = pynew(errcheck(@autopy x C.PyObject_ASCII(getptr(x_))))
pyascii(::Type{String}, x) = (s=pyascii(x); ans=pystr_asstring(s); pydel!(s); ans)
export pyascii

"""
    pyhasattr(x, k)

Equivalent to `hasattr(x, k)` in Python.

Tests if `getattr(x, k)` raises an `AttributeError`.
"""
function pyhasattr(x, k)
    ptr = @autopy x k C.PyObject_GetAttr(getptr(x_), getptr(k_))
    if iserrset(ptr)
        if errmatches(pybuiltins.AttributeError)
            errclear()
            return false
        else
            pythrow()
        end
    else
        decref(ptr)
        return true
    end
end
# pyhasattr(x, k) = errcheck(@autopy x k C.PyObject_HasAttr(getptr(x_), getptr(k_))) == 1
export pyhasattr

"""
    pygetattr(x, k, [d])

Equivalent to `getattr(x, k)` or `x.k` in Python.

If `d` is specified, it is returned if the attribute does not exist.
"""
pygetattr(x, k) = pynew(errcheck(@autopy x k C.PyObject_GetAttr(getptr(x_), getptr(k_))))
function pygetattr(x, k, d)
    ptr = @autopy x k C.PyObject_GetAttr(getptr(x_), getptr(k_))
    if iserrset(ptr)
        if errmatches(pybuiltins.AttributeError)
            errclear()
            return d
        else
            pythrow()
        end
    else
        return pynew(ptr)
    end
end
export pygetattr

"""
    pysetattr(x, k, v)

Equivalent to `setattr(x, k, v)` or `x.k = v` in Python.
"""
pysetattr(x, k, v) = (errcheck(@autopy x k v C.PyObject_SetAttr(getptr(x_), getptr(k_), getptr(v_))); nothing)
export pysetattr

"""
    pydelattr(x, k)

Equivalent to `delattr(x, k)` or `del x.k` in Python.
"""
pydelattr(x, k) = (errcheck(@autopy x k C.PyObject_SetAttr(getptr(x_), getptr(k_), C.PyNULL)); nothing)
export pydelattr

"""
    pyissubclass(s, t)

Test if `s` is a subclass of `t`. Equivalent to `issubclass(s, t)` in Python.
"""
pyissubclass(s, t) = errcheck(@autopy s t C.PyObject_IsSubclass(getptr(s_), getptr(t_))) == 1
export pyissubclass

"""
    pyisinstance(x, t)

Test if `x` is of type `t`. Equivalent to `isinstance(x, t)` in Python.
"""
pyisinstance(x, t) = errcheck(@autopy x t C.PyObject_IsInstance(getptr(x_), getptr(t_))) == 1
export pyisinstance

"""
    pyhash(x)

Equivalent to `hash(x)` in Python, converted to an `Integer`.
"""
pyhash(x) = errcheck(@autopy x C.PyObject_Hash(getptr(x_)))
export pyhash

"""
    pytruth(x)

The truthyness of `x`. Equivalent to `bool(x)` in Python, converted to a `Bool`.
"""
pytruth(x) = errcheck(@autopy x C.PyObject_IsTrue(getptr(x_))) == 1
export pytruth

"""
    pynot(x)

The falsyness of `x`. Equivalent to `not x` in Python, converted to a `Bool`.
"""
pynot(x) = errcheck(@autopy x C.PyObject_Not(getptr(x_))) == 1
export pynot

"""
    pylen(x)

The length of `x`. Equivalent to `len(x)` in Python, converted to an `Integer`.
"""
pylen(x) = errcheck(@autopy x C.PyObject_Length(getptr(x_)))
export pylen

"""
    pyhasitem(x, k)

Test if `pygetitem(x, k)` raises a `KeyError` or `AttributeError`.
"""
function pyhasitem(x, k)
    ptr = @autopy x k C.PyObject_GetItem(getptr(x_), getptr(k_))
    if iserrset(ptr)
        if errmatches(pybuiltins.KeyError) || errmatches(pybuiltins.IndexError)
            errclear()
            return false
        else
            pythrow()
        end
    else
        decref(ptr)
        return true
    end
end
export pyhasitem

"""
    pygetitem(x, k, [d])

Equivalent `x[k]` in Python.

If `d` is specified, it is returned if the item does not exist (i.e. if `x[k]` raises a
`KeyError` or `IndexError`).
"""
pygetitem(x, k) = pynew(errcheck(@autopy x k C.PyObject_GetItem(getptr(x_), getptr(k_))))
function pygetitem(x, k, d)
    ptr = @autopy x k C.PyObject_GetItem(getptr(x_), getptr(k_))
    if iserrset(ptr)
        if errmatches(pybuiltins.KeyError) || errmatches(pybuiltins.IndexError)
            errclear()
            return d
        else
            pythrow()
        end
    else
        return pynew(ptr)
    end
end
export pygetitem

"""
    pysetitem(x, k, v)

Equivalent to `setitem(x, k, v)` or `x[k] = v` in Python.
"""
pysetitem(x, k, v) = (errcheck(@autopy x k v C.PyObject_SetItem(getptr(x_), getptr(k_), getptr(v_))); nothing)
export pysetitem

"""
    pydelitem(x, k)

Equivalent to `delitem(x, k)` or `del x[k]` in Python.
"""
pydelitem(x, k) = (errcheck(@autopy x k C.PyObject_DelItem(getptr(x_), getptr(k_))); nothing)
export pydelitem

"""
    pydir(x)

Equivalent to `dir(x)` in Python.
"""
pydir(x) = pynew(errcheck(@autopy x C.PyObject_Dir(getptr(x_))))
export pydir

pycallargs(f) = pynew(errcheck(@autopy f C.PyObject_CallObject(getptr(f_), C.PyNULL)))
pycallargs(f, args) = pynew(errcheck(@autopy f args C.PyObject_CallObject(getptr(f_), getptr(args_))))
pycallargs(f, args, kwargs) = pynew(errcheck(@autopy f args kwargs C.PyObject_Call(getptr(f_), getptr(args_), getptr(kwargs_))))

"""
    pycall(f, args...; kwargs...)

Call the Python object `f` with the given arguments.
"""
pycall(f, args...; kwargs...) =
    if !isempty(kwargs)
        args_ = pytuple_fromiter(args)
        kwargs_ = pystrdict_fromiter(kwargs)
        ans = pycallargs(f, args_, kwargs_)
        pydel!(args_)
        pydel!(kwargs_)
        ans
    elseif !isempty(args)
        args_ = pytuple_fromiter(args)
        ans = pycallargs(f, args_)
        pydel!(args_)
        ans
    else
        pycallargs(f)
    end
export pycall

"""
    pyeq(x, y)
    pyeq(Bool, x, y)

Equivalent to `x == y` in Python. The second form converts to `Bool`.
"""
pyeq(x, y) = pynew(errcheck(@autopy x y C.PyObject_RichCompare(getptr(x_), getptr(y_), C.Py_EQ)))

"""
    pyne(x, y)
    pyne(Bool, x, y)

Equivalent to `x != y` in Python. The second form converts to `Bool`.
"""
pyne(x, y) = pynew(errcheck(@autopy x y C.PyObject_RichCompare(getptr(x_), getptr(y_), C.Py_NE)))

"""
    pyle(x, y)
    pyle(Bool, x, y)

Equivalent to `x <= y` in Python. The second form converts to `Bool`.
"""
pyle(x, y) = pynew(errcheck(@autopy x y C.PyObject_RichCompare(getptr(x_), getptr(y_), C.Py_LE)))

"""
    pylt(x, y)
    pylt(Bool, x, y)

Equivalent to `x < y` in Python. The second form converts to `Bool`.
"""
pylt(x, y) = pynew(errcheck(@autopy x y C.PyObject_RichCompare(getptr(x_), getptr(y_), C.Py_LT)))

"""
    pyge(x, y)
    pyge(Bool, x, y)

Equivalent to `x >= y` in Python. The second form converts to `Bool`.
"""
pyge(x, y) = pynew(errcheck(@autopy x y C.PyObject_RichCompare(getptr(x_), getptr(y_), C.Py_GE)))

"""
    pygt(x, y)
    pygt(Bool, x, y)

Equivalent to `x > y` in Python. The second form converts to `Bool`.
"""
pygt(x, y) = pynew(errcheck(@autopy x y C.PyObject_RichCompare(getptr(x_), getptr(y_), C.Py_GT)))
pyeq(::Type{Bool}, x, y) = errcheck(@autopy x y C.PyObject_RichCompareBool(getptr(x_), getptr(y_), C.Py_EQ)) == 1
pyne(::Type{Bool}, x, y) = errcheck(@autopy x y C.PyObject_RichCompareBool(getptr(x_), getptr(y_), C.Py_NE)) == 1
pyle(::Type{Bool}, x, y) = errcheck(@autopy x y C.PyObject_RichCompareBool(getptr(x_), getptr(y_), C.Py_LE)) == 1
pylt(::Type{Bool}, x, y) = errcheck(@autopy x y C.PyObject_RichCompareBool(getptr(x_), getptr(y_), C.Py_LT)) == 1
pyge(::Type{Bool}, x, y) = errcheck(@autopy x y C.PyObject_RichCompareBool(getptr(x_), getptr(y_), C.Py_GE)) == 1
pygt(::Type{Bool}, x, y) = errcheck(@autopy x y C.PyObject_RichCompareBool(getptr(x_), getptr(y_), C.Py_GT)) == 1
export pyeq, pyne, pyle, pylt, pyge, pygt

pyconvert_rule_object(::Type{Py}, x::Py) = pyconvert_return(x)

"""
    pycontains(x, v)

Equivalent to `v in x` in Python.
"""
pycontains(x, v) = errcheck(@autopy x v C.PySequence_Contains(getptr(x_), getptr(v_))) == 1
export pycontains

"""
    pyin(v, x)

Equivalent to `v in x` in Python.
"""
pyin(v, x) = pycontains(x, v)
export pyin

pynotin(v, x) = !pyin(v, x)
