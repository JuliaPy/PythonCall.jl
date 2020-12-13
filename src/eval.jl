function pyeval(::Type{R}, co::PyCode, globals::PyDict, locals::Union{PyDict,Nothing}=globals, extras::Union{NamedTuple,Nothing}=nothing) where {R}
    # get code
    cptr = pyptr(co)
    # get globals & ensure __builtins__ is set
    gptr = pyptr(globals)
    if !globals.hasbuiltins
        if C.PyMapping_HasKeyString(gptr, "__builtins__") == 0
            err = C.PyMapping_SetItemString(gptr, "__builtins__", C.PyEval_GetBuiltins())
            ism1(err) && pythrow()
        end
        globals.hasbuiltins = true
    end
    # get locals (ALLOCATES lptr if locals===nothing)
    if locals === nothing
        lptr = C.PyDict_New()
        isnull(lptr) && pythrow()
    else
        lptr = pyptr(locals)
    end
    # insert extra locals
    if extras !== nothing
        for (k,v) in pairs(extras)
            vo = C.PyObject_From(v)
            if isnull(vo)
                locals===nothing && C.Py_DecRef(lptr)
                pythrow()
            end
            err = C.PyMapping_SetItemString(lptr, string(k), vo)
            C.Py_DecRef(vo)
            if ism1(err)
                locals===nothing && C.Py_DecRef(lptr)
                pythrow()
            end
        end
    end
    # Call eval (ALLOCATES rptr)
    rptr = C.PyEval_EvalCode(cptr, gptr, lptr)
    if isnull(rptr)
        locals === nothing && C.Py_DecRef(lptr)
        pythrow()
    end
    # TODO: convert rptr using PyObject_As
    if co.mode == :exec
        if R <: Nothing
            C.Py_DecRef(rptr)
            locals===nothing && C.Py_DecRef(lptr)
            return nothing
        elseif R <: NamedTuple && isconcretetype(R)
            C.Py_DecRef(rptr)
            locals===nothing && C.Py_DecRef(lptr)
            error("returning NamedTuple not implemented yet")
        else
            C.Py_DecRef(rptr)
            locals===nothing && C.Py_DecRef(lptr)
            error("invalid return type $(R)")
        end
    elseif co.mode == :eval
        ret = C.PyObject_As(rptr, R)
        ret === PYUNCONVERTED() && C.PyErr_SetString(C.PyExc_TypeError(), "Cannot convert this '$(C.PyType_Name(C.Py_Type(rptr)))' to a Julia '$R'")
        C.Py_DecRef(rptr)
        locals===nothing && C.Py_DecRef(lptr)
        ret === PYERR() && pythrow()
        ret === PYUNCONVERTED() && pythrow()
        return ret
    else
        C.Py_DecRef(rptr)
        locals===nothing && C.Py_DecRef(lptr)
        error("invalid mode $(repr(co.mode))")
    end
end
export pyeval

module CompiledCode end

macro pyeval(R, code::String, locals=nothing)
    co = PyCode(code, "<julia $(__source__.file):$(__source__.line)>", :eval)
    nm = gensym()
    Base.eval(CompiledCode, :($nm = $co))
    :(pyeval($(esc(R)), $co, $(esc(:pyglobals)), $(esc(locals))))
end
export @pyeval

macro pyexec(code::String, locals=nothing)
    co = PyCode(code, "<julia $(__source__.file):$(__source__.line)>", :exec)
    nm = gensym()
    Base.eval(CompiledCode, :($nm = $co))
    :(pyeval(Nothing, $co, $(esc(:pyglobals)), $(esc(locals))))
end
export @pyexec
