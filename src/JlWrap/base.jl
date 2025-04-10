const pyjlbasetype = pynew()

_pyjl_getvalue(x) = @autopy x Cjl.PyJuliaValue_GetValue(getptr(x_))

_pyjl_setvalue!(x, v) = @autopy x Cjl.PyJuliaValue_SetValue(getptr(x_), v)

pyjl(t, v) = pynew(errcheck(@autopy t Cjl.PyJuliaValue_New(getptr(t_), v)))

"""
    pyisjl(x)

Test whether `x` is a wrapped Julia value, namely an instance of `juliacall.ValueBase`.
"""
pyisjl(x) = pytypecheck(x, pyjlbasetype)

pyjlisnull(x) = @autopy x begin
    if pyisjl(x_)
        Cjl.PyJuliaValue_IsNull(getptr(x_))
    else
        error("Expecting a 'juliacall.ValueBase', got a '$(pytype(x_).__name__)'")
    end
end

"""
    pyjlvalue(x)

Extract the value from the wrapped Julia value `x`.
"""
pyjlvalue(x) = @autopy x begin
    if pyjlisnull(x_)
        error("Julia value is NULL")
    else
        _pyjl_getvalue(x_)
    end
end

function init_base()
    setptr!(pyjlbasetype, incref(Cjl.PyJuliaBase_Type[]))
    pyjuliacallmodule.ValueBase = pyjlbasetype

    # conversion rule
    priority = PYCONVERT_PRIORITY_WRAP
    pyconvert_add_rule("juliacall:ValueBase", Any, pyconvert_rule_jlvalue, priority)
end

pyconvert_rule_jlvalue(::Type{T}, x::Py) where {T} =
    pyconvert_tryconvert(T, _pyjl_getvalue(x))

function Cjl._pyjl_callmethod(f, self_::C.PyPtr, args_::C.PyPtr, nargs::C.Py_ssize_t)
    @nospecialize f
    if Cjl.PyJuliaValue_IsNull(self_)
        errset(pybuiltins.TypeError, "Julia object is NULL")
        return C.PyNULL
    end
    in_f = false
    self = Cjl.PyJuliaValue_GetValue(self_)
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
        elseif nargs == 3
            arg1 = pynew(incref(C.PyTuple_GetItem(args_, 1)))
            arg2 = pynew(incref(C.PyTuple_GetItem(args_, 2)))
            in_f = true
            ans = f(self, arg1, arg2)::Py
            in_f = false
        elseif nargs == 4
            arg1 = pynew(incref(C.PyTuple_GetItem(args_, 1)))
            arg2 = pynew(incref(C.PyTuple_GetItem(args_, 2)))
            arg3 = pynew(incref(C.PyTuple_GetItem(args_, 3)))
            in_f = true
            ans = f(self, arg1, arg2, arg3)::Py
            in_f = false
        else
            errset(
                pybuiltins.NotImplementedError,
                "__jl_callmethod not implemented for this many arguments",
            )
        end
        return incref(getptr(ans))
    catch exc
        if exc isa PyException
            Base.GC.@preserve exc C.PyErr_Restore(
                incref(getptr(exc._t)),
                incref(getptr(exc._v)),
                incref(getptr(exc._b)),
            )
            return C.PyNULL
        else
            try
                if in_f
                    return pyjl_handle_error(f, self, exc)
                else
                    errset(
                        pyJuliaError,
                        pytuple((pyjlraw(exc), pyjlraw(catch_backtrace()))),
                    )
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
    if pyisnull(t)
        # NULL => raise JuliaError
        errset(pyJuliaError, pytuple((pyjlraw(exc), pyjlraw(catch_backtrace()))))
        return C.PyNULL
    elseif pyistype(t)
        # Exception type => raise this type of error
        errset(t, string("Julia: ", Py(sprint(showerror, exc))))
        return C.PyNULL
    else
        # Otherwise, return the given object (e.g. NotImplemented)
        return Base.GC.@preserve t incref(getptr(t))
    end
end

function pyjl_methodnum(f)
    @nospecialize f
    Cjl.PyJulia_MethodNum(f)
end

function pyjl_handle_error_type(f, self, exc)
    @nospecialize f self exc
    PyNULL
end

Py(x) = ispy(x) ? throw(MethodError(Py, (x,))) : pyjl(x)
