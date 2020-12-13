mutable struct PyDict
    ptr :: CPyPtr
    hasbuiltins :: Bool
    function PyDict()
        o = new(CPyPtr(), false)
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
export PyDict

const pyglobals = PyDict()
export pyglobals

function pyptr(x::PyDict)
    ptr = getfield(x, :ptr)
    if isnull(ptr)
        ptr = C.PyDict_New()
        if isnull(ptr)
            pythrow()
        else
            setfield!(x, :ptr, ptr)
            ptr
        end
    else
        ptr
    end
end
