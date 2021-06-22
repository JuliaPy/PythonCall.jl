pyis(x, y) = @autopy x y getptr(x_) == getptr(y_)
export pyis

pyrepr(x) = pynew(errcheck(@autopy x C.PyObject_Repr(getptr(x_))))
pyrepr(::Type{String}, x) = (s=pyrepr(x); ans=pystr_asstring(s); pydone!(s); ans)
export pyrepr

pyascii(x) = pynew(errcheck(@autopy x C.PyObject_ASCII(getptr(x_))))
pyascii(::Type{String}, x) = (s=pyascii(x); ans=pystr_asstring(s); pydone!(s); ans)
export pyascii

pyhasattr(x, k) = errcheck(@autopy x k C.PyObject_HasAttr(getptr(x_), getptr(k_))) == 1
export pyhasattr

pygetattr(x, k) = pynew(errcheck(@autopy x k C.PyObject_GetAttr(getptr(x_), getptr(k_))))
export pygetattr

pysetattr(x, k, v) = (errcheck(@autopy x k v C.PyObject_SetAttr(getptr(x_), getptr(k_), getptr(v_))); nothing)
export pysetattr

pydelattr(x, k) = (errcheck(@autopy x k C.PyObject_SetAttr(getptr(x_), getptr(k_), C.PyNULL)); nothing)
export pydelattr

pyissubclass(s, t) = errcheck(@autopy s t C.PyObject_IsSubclass(getptr(s_), getptr(t_))) == 1
export pyissubclass

pyisinstance(x, t) = errcheck(@autopy x t C.PyObject_IsInstance(getptr(x_), getptr(t_))) == 1
export pyisinstance

pyhash(x) = errcheck(@autopy x C.PyObject_Hash(getptr(x_)))
export pyhash

pytruth(x) = errcheck(@autopy x C.PyObject_IsTrue(getptr(x_))) == 1
export pytruth

pynot(x) = errcheck(@autopy x C.PyObject_IsNot(getptr(x_))) == 1
export pynot

pylen(x) = errcheck(@autopy x C.PyObject_Length(getptr(x_)))
export pylen

pygetitem(x, k) = pynew(errcheck(@autopy x k C.PyObject_GetItem(getptr(x_), getptr(k_))))
export pygetitem

pysetitem(x, k, v) = (errcheck(@autopy x k v C.PyObject_SetItem(getptr(x_), getptr(k_), getptr(v_))); nothing)
export pysetitem

pydelitem(x, k) = (errcheck(@autopy x k C.PyObject_DelItem(getptr(x_), getptr(k_))); nothing)
export pydelitem

pydir(x) = pynew(errcheck(@autopy x C.PyObject_Dir(getptr(x_))))
export pydir

pycallargs(f) = pynew(errcheck(@autopy f C.PyObject_CallObject(getptr(f_), C.PyNULL)))
pycallargs(f, args) = pynew(errcheck(@autopy f args C.PyObject_CallObject(getptr(f_), getptr(args_))))
pycallargs(f, args, kwargs) = pynew(errcheck(@autopy f args kwargs C.PyObject_Call(getptr(f_), getptr(args_), getptr(kwargs_))))

pycall(f, args...; kwargs...) =
    if !isempty(kwargs)
        args_ = pytuple_fromiter(args)
        kwargs_ = pystrdict_fromiter(kwargs)
        ans = pycallargs(f, args_, kwargs_)
        pydone!(args_)
        pydone!(kwargs_)
        ans
    elseif !isempty(args)
        args_ = pytuple_fromiter(args)
        ans = pycallargs(f, args_)
        pydone!(args_)
        ans
    else
        pycallargs(f)
    end
export pycall

pyeq(x, y) = pynew(errcheck(@autopy x y C.PyObject_RichCompare(getptr(x_), getptr(y_), C.Py_EQ)))
pyne(x, y) = pynew(errcheck(@autopy x y C.PyObject_RichCompare(getptr(x_), getptr(y_), C.Py_NE)))
pyle(x, y) = pynew(errcheck(@autopy x y C.PyObject_RichCompare(getptr(x_), getptr(y_), C.Py_LE)))
pylt(x, y) = pynew(errcheck(@autopy x y C.PyObject_RichCompare(getptr(x_), getptr(y_), C.Py_LT)))
pyge(x, y) = pynew(errcheck(@autopy x y C.PyObject_RichCompare(getptr(x_), getptr(y_), C.Py_GE)))
pygt(x, y) = pynew(errcheck(@autopy x y C.PyObject_RichCompare(getptr(x_), getptr(y_), C.Py_GT)))
pyeq(::Type{Bool}, x, y) = errcheck(@autopy x y C.PyObject_RichCompareBool(getptr(x_), getptr(y_), C.Py_EQ)) == 1
pyne(::Type{Bool}, x, y) = errcheck(@autopy x y C.PyObject_RichCompareBool(getptr(x_), getptr(y_), C.Py_NE)) == 1
pyle(::Type{Bool}, x, y) = errcheck(@autopy x y C.PyObject_RichCompareBool(getptr(x_), getptr(y_), C.Py_LE)) == 1
pylt(::Type{Bool}, x, y) = errcheck(@autopy x y C.PyObject_RichCompareBool(getptr(x_), getptr(y_), C.Py_LT)) == 1
pyge(::Type{Bool}, x, y) = errcheck(@autopy x y C.PyObject_RichCompareBool(getptr(x_), getptr(y_), C.Py_GE)) == 1
pygt(::Type{Bool}, x, y) = errcheck(@autopy x y C.PyObject_RichCompareBool(getptr(x_), getptr(y_), C.Py_GT)) == 1
export pyeq, pyne, pyle, pylt, pyge, pygt

pyconvert_rule_object(::Type{Py}, x::Py) = x
