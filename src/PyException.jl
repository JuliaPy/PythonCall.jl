mutable struct PyException <: Exception
    tptr :: CPyPtr
    vptr :: CPyPtr
    bptr :: CPyPtr
    function PyException(::Val{:new}, t::Ptr=C_NULL, v::Ptr=C_NULL, b::Ptr=C_NULL, borrowed::Bool=false)
        o = new(CPyPtr(t), CPyPtr(v), CPyPtr(b))
        if borrowed
            C.Py_IncRef(t)
            C.Py_IncRef(v)
            C.Py_IncRef(b)
        end
        finalizer(o) do o
            if CONFIG.isinitialized
                tptr = getfield(o, :tptr)
                vptr = getfield(o, :vptr)
                bptr = getfield(o, :bptr)
                if !isnull(tptr) || !isnull(vptr) || !isnull(bptr)
                    with_gil(false) do
                        C.Py_DecRef(tptr)
                        C.Py_DecRef(vptr)
                        C.Py_DecRef(bptr)
                    end
                end
            end
        end
        return o
    end
end
export PyException

pythrow() = throw(PyException(Val(:new), C.PyErr_FetchTuple()...))

function Base.showerror(io::IO, e::PyException)
    if isnull(e.tptr)
        print(io, "Python: mysterious error (no error was actually set)")
        return
    end

    if CONFIG.sysautolasttraceback
        err = C.Py_DecRef(C.PyImport_ImportModule("sys"), PYERR()) do sys
            err = C.PyObject_SetAttrString(sys, "last_type", isnull(e.tptr) ? C.Py_None() : e.tptr)
            ism1(err) && return PYERR()
            err = C.PyObject_SetAttrString(sys, "last_value", isnull(e.vptr) ? C.Py_None() : e.vptr)
            ism1(err) && return PYERR()
            err = C.PyObject_SetAttrString(sys, "last_traceback", isnull(e.bptr) ? C.Py_None() : e.bptr)
            ism1(err) && return PYERR()
            nothing
        end
        if err == PYERR()
            C.PyErr_Clear()
            print(io, "<error while setting 'sys.last_traceback'>")
        end
    end

    # # if this is a Julia exception then recursively print it and its stacktrace
    # if pyerrmatches(e.t, pyjlexception)
    #     try
    #         jp = pyjlgetvalue(e.v.args[0])
    #         if jp !== nothing
    #             je, jb = jp
    #             print(io, "Python: Julia: ")
    #             showerror(io, je)
    #             if jb === nothing
    #                 println(io)
    #                 print(io, "Stacktrace: none")
    #             else
    #                 io2 = IOBuffer()
    #                 Base.show_backtrace(IOContext(io2, :color=>true, :displaysize=>displaysize(io)), jb)
    #                 printstyled(io, String(take!(io2)))
    #             end
    #         end
    #         if pyisnone(e.b)
    #             println(io)
    #             printstyled(io, "Python stacktrace: none")
    #             return
    #         else
    #             @goto pystacktrace
    #         end
    #     catch err
    #         println(io, "<error while printing Julia excpetion inside Python exception: $(err)>")
    #     end
    # end

    # otherwise, print the Python exception ****** TODO ******
    print(io, "Python: ")

    # print the type name
    tname = C.Py_DecRef(C.PyObject_GetAttrString(e.tptr, "__name__")) do tnameo
        C.PyUnicode_As(tnameo, String)
    end
    if tname === PYERR()
        C.PyErr_Clear()
        print(io, "<error while printing type>")
    else
        print(io, tname)
    end

    # print the error message
    if !isnull(e.vptr)
        print(io, ": ")
        vstr = C.PyObject_StrAs(e.vptr, String)
        if vstr === PYERR()
            C.PyErr_Clear()
            print(io, "<error while printing value>")
        else
            print(io, vstr)
        end
    end

    # print the stacktrace
    if !isnull(e.bptr)
        @label pystacktrace
        println(io)
        printstyled(io, "Python stacktrace:")
        err = C.Py_DecRef(C.PyImport_ImportModule("traceback")) do tb
            C.Py_DecRef(C.PyObject_GetAttrString(tb, "extract_tb")) do extr
                C.Py_DecRef(C.PyObject_CallNice(extr, C.PyObjectRef(e.bptr))) do fs
                    nfs = C.PySequence_Length(fs)
                    ism1(nfs) && return PYERR()
                    for i in 1:nfs
                        println(io)
                        printstyled(io, " [", i, "] ")
                        # name
                        err = C.Py_DecRef(C.PySequence_GetItem(fs, i-1)) do f
                            name = C.Py_DecRef(C.PyObject_GetAttrString(f, "name")) do nameo
                                C.PyObject_StrAs(nameo, String)
                            end
                            name === PYERR() && return PYERR()
                            printstyled(io, name, bold=true)
                            printstyled(io, " at ")
                            fname = C.Py_DecRef(C.PyObject_GetAttrString(f, "filename")) do fnameo
                                C.PyObject_StrAs(fnameo, String)
                            end
                            fname === PYERR() && return PYERR()
                            printstyled(io, fname, ":", bold=true)
                            lineno = C.Py_DecRef(C.PyObject_GetAttrString(f, "lineno")) do linenoo
                                C.PyObject_StrAs(linenoo, String)
                            end
                            lineno === PYERR() && return PYERR()
                            printstyled(io, lineno, bold=true)
                            nothing
                        end
                        err === PYERR() && return PYERR()
                    end
                end
            end
        end
        if err === PYERR()
            C.PyErr_Clear()
            print(io, "<error while printing stacktrace>")
        end
    end
end
