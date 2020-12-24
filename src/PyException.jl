mutable struct PyException <: Exception
    tref :: PyRef
    vref :: PyRef
    bref :: PyRef
    PyException(::Val{:new}, t::Ptr=C_NULL, v::Ptr=C_NULL, b::Ptr=C_NULL, borrowed::Bool=false) =
        new(PyRef(Val(:new), t, borrowed), PyRef(Val(:new), v, borrowed), PyRef(Val(:new), b, borrowed))
end
export PyException

pythrow() = throw(PyException(Val(:new), C.PyErr_FetchTuple(true)...))

"""
    check(x)

Checks if the Python error indicator is set. If so, throws the current Python exception. Otherwise returns `x`.
"""
check(x) = C.PyErr_IsSet() ? pythrow() : x

"""
    checkm1(x, ambig=false)

Same as `check(x)` but errors are indicated by `x == -1` instead.

If `ambig` is true the error indicator is also checked.
"""
checkm1(x::Number, ambig::Bool=false) = ism1(x) ? ambig ? check(x) : pythrow() : x

"""
    checknull(x, ambig=false)

Same as `check(x)` but errors are indicated by `x == C_NULL` instead.

If `ambig` is true the error indicator is also checked.
"""
checknull(x::Ptr, ambig::Bool=false) = isnull(x) ? ambig ? check(x) : pythrow() : x

"""
    checkerr(x, ambig=false)

Same as `check(x)` but errors are indicated by `x == PYERR()` instead.

If `ambig` is true the error indicator is also checked.
"""
checkerr(x, ambig::Bool=false) = x===PYERR() ? ambig ? check(x) : pythrow() : x

"""
    checknullconvert(T, x, ambig=false) :: T

Same as `checknull(x, ambig)` but steals a reference to `x` and converts the result to a `T`.
"""
checknullconvert(::Type{T}, x::Ptr, ambig::Bool=false) where {T} = begin
    if isnull(x) && (!ambig || C.PyErr_IsSet())
        C.Py_DecRef(x)
        pythrow()
    end
    r = C.PyObject_Convert(x, T)
    C.Py_DecRef(x)
    checkm1(r)
    C.takeresult(T)
end

function Base.showerror(io::IO, e::PyException)
    if isnull(e.tref)
        print(io, "Python: mysterious error (no error was actually set)")
        return
    end

    if CONFIG.sysautolasttraceback
        err = C.Py_DecRef(C.PyImport_ImportModule("sys"), PYERR()) do sys
            err = C.PyObject_SetAttrString(sys, "last_type", isnull(e.tref) ? C.Py_None() : e.tref.ptr)
            ism1(err) && return PYERR()
            err = C.PyObject_SetAttrString(sys, "last_value", isnull(e.vref) ? C.Py_None() : e.vref.ptr)
            ism1(err) && return PYERR()
            err = C.PyObject_SetAttrString(sys, "last_traceback", isnull(e.bref) ? C.Py_None() : e.bref.ptr)
            ism1(err) && return PYERR()
            nothing
        end
        if err == PYERR()
            C.PyErr_Clear()
            print(io, "<error while setting 'sys.last_traceback'>")
        end
    end

    # if this is a Julia exception then recursively print it and its stacktrace
    if C.PyErr_GivenExceptionMatches(e.tref, C.PyExc_JuliaError()) != 0
        try
            # Extract error value
            vo = @pyv `$(e.vref).args[0]`::Any
            if vo isa Tuple{Exception, Any}
                je, jb = vo
            else
                je = vo
                jb = nothing
            end
            print(io, "Python: Julia: ")
            # Show exception
            if je isa Exception
                showerror(io, je)
            else
                print(io, je)
            end
            # Show backtrace
            if jb === nothing
                println(io)
                print(io, "Stacktrace: none")
            else
                io2 = IOBuffer()
                Base.show_backtrace(IOContext(io2, :color=>true, :displaysize=>displaysize(io)), jb)
                printstyled(io, String(take!(io2)))
            end
            # Show Python backtrace
            if isnull(e.bref)
                println(io)
                printstyled(io, "Python stacktrace: none")
                return
            else
                @goto pystacktrace
            end
        catch err
            println("<error while printing Julia exception inside Python exception>")
        end
    end

    # otherwise, print the Python exception
    print(io, "Python: ")

    # print the type name
    try
        tname = @pyv `$(e.tref).__name__`::String
        print(io, tname)
    catch
        print(io, "<error while printing type>")
    end

    # print the error message
    if !isnull(e.vref)
        print(io, ": ")
        try
            vstr = @pyv `str($(e.vref))`::String
            print(io, vstr)
        catch
            print(io, "<error while printing value>")
        end
    end

    # print the stacktrace
    if !isnull(e.bref)
        @label pystacktrace
        println(io)
        printstyled(io, "Python stacktrace:")
        try
            @py ```
            import traceback
            $(fs :: Vector{Tuple{String, String, Int}}) = [(x.name, x.filename, x.lineno) for x in traceback.extract_tb($(e.bref))]
            ```
            for (i,(name, fname, lineno)) in enumerate(fs)
                println(io)
                printstyled(io, " [", i, "] ")
                printstyled(io, name, bold=true)
                printstyled(io, " at ")
                # if (m=match(r"^(.*):([0-9]+)$", fname)) !== nothing
                #     fname = m.captures[1]
                #     lineno += parse(Int, m.captures[2]) - 1
                # end
                printstyled(io, fname, ":", lineno, bold=true)
            end
        catch err
            print(io, "<error while printing stacktrace: $err>")
        end
    end
end
