mutable struct PyCode
    ptr :: CPyPtr
    code :: String
    filename :: String
    mode :: Symbol
    function PyCode(code::String, filename::String, mode::Symbol)
        mode in (:exec, :eval) || error("invalid mode $(repr(mode))")
        o = new(CPyPtr(), code, filename, mode)
        finalizer(o) do o
            if CONFIG.isinitialized
                ptr = getfield(o, :ptr)
                if !isnull(ptr)
                    with_gil(false) do
                        C.Py_DecRef(ptr)
                    end
                end
            end
        end
        return o
    end
end
export PyCode

function pyptr(co::PyCode)
    ptr = getfield(co, :ptr)
    if isnull(ptr)
        ptr = C.Py_CompileString(co.code, co.filename, co.mode == :exec ? C.Py_file_input : co.mode == :eval ? C.Py_eval_input : error("invalid mode $(repr(co.mode))"))
        if isnull(ptr)
            pythrow()
        else
            setfield!(co, :ptr, ptr)
            ptr
        end
    else
        ptr
    end
end
