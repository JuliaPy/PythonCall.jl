const pyjlbasetype = pynew()

_pyjl_getvalue(x) = @autopy x C.PyJuliaValue_GetValue(getptr(x_))

_pyjl_setvalue!(x, v) = @autopy x C.PyJuliaValue_SetValue(getptr(x_), v)

pyjl(t, v) = pynew(errcheck(@autopy t C.PyJuliaValue_New(getptr(t_), v)))
export pyjl

pyisjl(x) = pytypecheck(x, pyjlbasetype)
export pyisjl

pyjlvalue(x) = @autopy x begin
    if pyisjl(x_)
        _pyjl_getvalue(x_)
    else
        error("Expecting a 'juliacall.ValueBase', got a '$(pytype(x_).__name__)'")
    end
end
export pyjlvalue

function init_jlwrap_base()
    setptr!(pyjlbasetype, incref(C.POINTERS.PyJuliaBase_Type))
    pyjuliacallmodule.ValueBase = pyjlbasetype
end

pyconvert_rule_jlvalue(::Type{T}, x::Py) where {T} = pyconvert_tryconvert(T, _pyjl_getvalue(x))

function C._pyjl_callmethod(f::Function, self_::C.PyPtr, args_::C.PyPtr, nargs::C.Py_ssize_t)
    @nospecialize f
    in_f = false
    self = C.PyJuliaValue_GetValue(self_)
    try
        if nargs == 1
            in_f = true
            ans = f(self)
            in_f = false
        elseif nargs == 2
            arg1 = pynew(incref(C.PyTuple_GetItem(args_, 1)))
            in_f = true
            ans = f(self, arg1)
            in_f = false
            pydel!(arg1)
        elseif nargs == 3
            arg1 = pynew(incref(C.PyTuple_GetItem(args_, 1)))
            arg2 = pynew(incref(C.PyTuple_GetItem(args_, 2)))
            in_f = true
            ans = f(self, arg1, arg2)
            in_f = false
            pydel!(arg1)
            pydel!(arg2)
        elseif nargs == 4
            arg1 = pynew(incref(C.PyTuple_GetItem(args_, 1)))
            arg2 = pynew(incref(C.PyTuple_GetItem(args_, 2)))
            arg3 = pynew(incref(C.PyTuple_GetItem(args_, 3)))
            in_f = true
            ans = f(self, arg1, arg2, arg3)
            in_f = false
            pydel!(arg1)
            pydel!(arg2)
            pydel!(arg3)
        else
            errset(pybuiltins.NotImplementedError, "__jl_callmethod not implemented for this many arguments")
        end
        ptr = getptr(ans::Py)
        pystolen!(ans)
        return ptr
    catch exc
        if exc isa PyException
            C.PyErr_Restore(incref(getptr(exc._t)), incref(getptr(exc._v)), incref(getptr(exc._b)))
        else
            try
                if in_f
                    pyjl_handle(f, self, exc)
                else
                    errset(pyJuliaError, Py(pyjlraw((exc, catch_backtrace()))))
                end
            catch
                errset(pyJuliaError, "an error occurred while setting an error")
            end
        end
        return C.PyNULL
    end
end

function pyjl_handle(f, self, exc)
    @nospecialize f self exc
    t = pyjl_handle_type(f, self, exc)
    if ispynull(t)
        errset(pyJuliaError, Py(pyjlraw((exc, catch_backtrace()))))
    else
        errset(t, Py(sprint(showerror, exc)))
    end
end

pyjl_methodnum(@nospecialize(f)) = C.PyJulia_MethodNum(f)

macro pyjlmethods_str(s)
    replace(s, r"\$[a-zA-Z0-9_]+\(" => x -> string("_jl_callmethod(", pyjl_methodnum(__module__.eval(Symbol(x[2:end-1]))), ", "))
end

pyjl_handle_type(f, self, exc) = pynew()
