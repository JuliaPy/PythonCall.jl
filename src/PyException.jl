"""
    PyException <: Exception

Represents an exception raised from Python.

It has three fields `tref`, `vref`, `bref` which are all `PyRef`s, and are the type, value and backtrace of the exception.
"""
mutable struct PyException <: Exception
    tref::PyRef
    vref::PyRef
    bref::PyRef
    PyException(
        ::Val{:new},
        t::Ptr = C_NULL,
        v::Ptr = C_NULL,
        b::Ptr = C_NULL,
        borrowed::Bool = false,
    ) = new(
        PyRef(Val(:new), t, borrowed),
        PyRef(Val(:new), v, borrowed),
        PyRef(Val(:new), b, borrowed),
    )
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
checkm1(x::Number, ambig::Bool = false) = ism1(x) ? ambig ? check(x) : pythrow() : x

"""
    checknull(x, ambig=false)

Same as `check(x)` but errors are indicated by `x == C_NULL` instead.

If `ambig` is true the error indicator is also checked.
"""
checknull(x::Ptr, ambig::Bool = false) = isnull(x) ? ambig ? check(x) : pythrow() : x

"""
    checkerr(x, ambig=false)

Same as `check(x)` but errors are indicated by `x == PYERR()` instead.

If `ambig` is true the error indicator is also checked.
"""
checkerr(x, ambig::Bool = false) = x === PYERR() ? ambig ? check(x) : pythrow() : x

"""
    checknullconvert(T, x, ambig=false) :: T

Same as `checknull(x, ambig)` but steals a reference to `x` and converts the result to a `T`.
"""
checknullconvert(::Type{T}, x::Ptr, ambig::Bool = false) where {T} = begin
    if isnull(x) && (!ambig || C.PyErr_IsSet())
        C.Py_DecRef(x)
        pythrow()
    end
    r = C.PyObject_Convert(x, T)
    C.Py_DecRef(x)
    checkm1(r)
    C.takeresult(T)
end

file_to_pymodule(fname::String) = begin
    isfile(fname) || return nothing
    modules = PyDict{String}(pyimport("sys").modules)
    for (n,m) in modules
        if pyhasattr(m, :__file__)
            fname2 = pystr(String, m.__file__)
            if isfile(fname2) && realpath(fname) == realpath(fname2)
                return n
            end
        end
    end
end

function Base.showerror(io::IO, e::PyException)
    print(io, "Python: ")

    if isnull(e.tref)
        print(io, "mysterious error (no error was actually set)")
        return
    end

    if CONFIG.sysautolasttraceback
        try
            @py ```
            import sys
            sys.last_type = $(isnull(e.tref) ? nothing : e.tref)
            sys.last_value = $(isnull(e.vref) ? nothing : e.vref)
            sys.last_traceback = $(isnull(e.bref) ? nothing : e.bref)
            ```
        catch err
            print(io, "<error while setting 'sys.last_traceback': $err")
        end
    end

    # if this is a Julia exception then recursively print it and its stacktrace
    if C.PyErr_GivenExceptionMatches(e.tref, C.PyExc_JuliaError()) != 0
        try
            # Extract error value
            vo = @pyv `$(e.vref).args[0]`::Any
            if vo isa Tuple{Exception,Any}
                je, jb = vo
            else
                je = vo
                jb = nothing
            end
            print(io, "Julia: ")
            # Show exception
            if je isa Exception
                showerror(io, je)
            else
                print(io, je)
            end
            # Show backtrace
            if jb === nothing
                println(io)
                printstyled(io, "Stacktrace: none")
            else
                io2 = IOBuffer()
                Base.show_backtrace(
                    IOContext(io2, io),
                    jb,
                )
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
            println("<error while printing Julia exception inside Python exception: $err>")
        end
    end

    # print the type name
    try
        tname = @pyv `$(e.tref).__name__`::String
        print(io, tname)
    catch err
        print(io, "<error while printing type: $err>")
    end

    # print the error message
    if !isnull(e.vref)
        print(io, ": ")
        try
            vstr = @pyv `str($(e.vref))`::String
            print(io, vstr)
        catch err
            print(io, "<error while printing value: $err>")
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
            if VERSION < v"1.6.0-rc1"
                for (i, (name, fname, lineno)) in enumerate(reverse(fs))
                    println(io)
                    printstyled(io, " [", i, "] ")
                    printstyled(io, name, bold = true)
                    printstyled(io, " at ")
                    printstyled(io, fname, ":", lineno, bold = true)
                end
            else
                mcdict = Dict{String, Symbol}()
                mccyclyer = Iterators.Stateful(Iterators.cycle(Base.STACKTRACE_MODULECOLORS))
                # skip a couple as a basic attempt to make the colours different from the Julia stacktrace
                popfirst!(mccyclyer)
                popfirst!(mccyclyer)
                for (i, (name, fname, lineno)) in enumerate(reverse(fs))
                    println(io)
                    printstyled(io, " [", i, "] ")
                    printstyled(io, name, bold = true)
                    println(io)
                    printstyled(io, "   @ ", color = :light_black)
                    mod = file_to_pymodule(fname)
                    if mod !== nothing
                        # print the module, with colour determined by the top level name
                        tmod = first(split(mod, ".", limit=2))
                        color = get!(mcdict, tmod) do
                            popfirst!(mccyclyer)
                        end
                        printstyled(io, mod, " ", color = color)
                    end
                    if isfile(fname) && Base.stacktrace_contract_userdir()
                        fname = Base.replaceuserpath(fname)
                    end
                    printstyled(io, fname, ":", lineno, color = :light_black)
                end
            end
        catch err
            print(io, "<error while printing stacktrace: $err>")
        end
    end
end
