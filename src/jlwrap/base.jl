const pyjlbasetype = pynew()

_pyjl_getvalue(x) = @autopy x C.PyJuliaValue_GetValue(getptr(x_))

_pyjl_setvalue!(x, v) = @autopy x C.PyJuliaValue_SetValue(getptr(x_), v)

pyjl(t, v) = pynew(errcheck(@autopy t C.PyJuliaValue_New(getptr(t_), v)))
export pyjl

pyisjl(x) = pytypecheck(x, pyjlbasetype)
export pyisjl

pyjlisnull(x) = @autopy x begin
    if pyisjl(x_)
        C.PyJuliaValue_IsNull(getptr(x_))
    else
        error("Expecting a 'juliacall.ValueBase', got a '$(pytype(x_).__name__)'")
    end
end

pyjlvalue(x) = @autopy x begin
    if pyjlisnull(x_)
        error("Julia value is NULL")
    else
        _pyjl_getvalue(x_)
    end
end
export pyjlvalue

function init_jlwrap_base()
    setptr!(pyjlbasetype, incref(C.POINTERS.PyJuliaBase_Type))
    pyjuliacallmodule.ValueBase = pyjlbasetype
end

pyconvert_rule_jlvalue(::Type{T}, x::Py) where {T} = pyconvert_tryconvert(T, _pyjl_getvalue(x))

function C._pyjl_callmethod(f, self_::C.PyPtr, args_::C.PyPtr, nargs::C.Py_ssize_t)
    @nospecialize f
    if C.PyJuliaValue_IsNull(self_)
        errset(pybuiltins.TypeError, "Julia object is NULL")
        return C.PyNULL
    end
    in_f = false
    self = C.PyJuliaValue_GetValue(self_)
    try
        if nargs == 1
            in_f = true
            ans = f(self)::Py
            in_f = false
        elseif nargs == 2
            arg1 = pynew(incref(C.PyTuple_GetItem(args_, 1)))
            in_f = true
            ans = f(self, arg1)::Py
            in_f = false
            pydel!(arg1)
        elseif nargs == 3
            arg1 = pynew(incref(C.PyTuple_GetItem(args_, 1)))
            arg2 = pynew(incref(C.PyTuple_GetItem(args_, 2)))
            in_f = true
            ans = f(self, arg1, arg2)::Py
            in_f = false
            pydel!(arg1)
            pydel!(arg2)
        elseif nargs == 4
            arg1 = pynew(incref(C.PyTuple_GetItem(args_, 1)))
            arg2 = pynew(incref(C.PyTuple_GetItem(args_, 2)))
            arg3 = pynew(incref(C.PyTuple_GetItem(args_, 3)))
            in_f = true
            ans = f(self, arg1, arg2, arg3)::Py
            in_f = false
            pydel!(arg1)
            pydel!(arg2)
            pydel!(arg3)
        else
            errset(pybuiltins.NotImplementedError, "__jl_callmethod not implemented for this many arguments")
        end
        ptr = getptr(ans)
        pystolen!(ans)
        return ptr
    catch exc
        if exc isa PyException
            C.PyErr_Restore(incref(getptr(exc._t)), incref(getptr(exc._v)), incref(getptr(exc._b)))
            return C.PyNULL
        else
            try
                if in_f
                    return pyjl_handle_error(f, self, exc)
                else
                    errset(pyJuliaError, pyjlraw((exc, catch_backtrace())))
                    return C.PyNULL
                end
            catch
                errset(pyJuliaError, "an error occurred while setting an error")
                return C.PyNULL
            end
        end
    end
end

function pyjl_handle_error(f, self, exc)
    @nospecialize f self exc
    t = pyjl_handle_error_type(f, self, exc)::Py
    if ispynull(t)
        # NULL => raise JuliaError
        errset(pyJuliaError, pyjlraw((exc, catch_backtrace())))
        return C.PyNULL
    elseif pyistype(t)
        # Exception type => raise this type of error
        errset(t, string("Julia: ", Py(sprint(showerror, exc))))
        return C.PyNULL
    else
        # Otherwise, return the given object (e.g. NotImplemented)
        return incref(getptr(t))
    end
end

function pyjl_methodnum(f)
    @nospecialize f
    C.PyJulia_MethodNum(f)
end

function pyjl_handle_error_type(f, self, exc)
    @nospecialize f self exc
    PyNULL
end
