check(::Type{CPyPtr}, o::CPyPtr, ambig=false) = (o == C_NULL && (ambig ? pyerrcheck() : pythrow()); o)
check(::Type{T}, o::T, ambig=false) where {T<:Number} = (o == (zero(T)-one(T)) && (ambig ? pyerrcheck() : pythrow()); o)
check(::Type{Nothing}, o::Cvoid, ambig=false) = ambig ? pyerrcheck() : nothing
check(::Type{PyObject}, o::CPyPtr, ambig=false) = pynewobject(check(CPyPtr, o, ambig), false)
check(::Type{Bool}, o::Cint, ambig=false) = !iszero(check(Cint, o, ambig))
check(::Type{Nothing}, o::Cint, ambig=false) = (check(Cint, o, ambig); nothing)
check(o::CPyPtr, ambig=false) = check(PyObject, o, ambig)
check(o::T, ambig=false) where {T<:Number} = check(T, o, ambig)
check(o::Cvoid, ambig=false) = check(Nothing, o, ambig)


pyerroccurred() = C.PyErr_Occurred() != C_NULL

function pyerroccurred(t::AbstractPyObject)
    e = C.PyErr_Occurred()
    e != C_NULL && !iszero(C.PyErr_GivenExceptionMatches(e, t))
end

pyerrclear() = C.PyErr_Clear()

pyerrcheck() = pyerroccurred() ? pythrow() : nothing

pyerrset(t::AbstractPyObject) = C.PyErr_SetNone(t)
pyerrset(t::AbstractPyObject, x::AbstractString) = C.PyErr_SetString(t, x)
pyerrset(t::AbstractPyObject, x::AbstractPyObject) = C.PyErr_SetObject(t, x)

pyerrmatches(e::AbstractPyObject, t::AbstractPyObject) = !iszero(C.PyErr_GivenExceptionMatches(e, t))

struct PythonRuntimeError <: Exception
    t :: PyObject
    v :: PyObject
    b :: PyObject
end

function pythrow()
    t = Ref{CPyPtr}()
    v = Ref{CPyPtr}()
    b = Ref{CPyPtr}()
    C.PyErr_Fetch(t, v, b)
    C.PyErr_NormalizeException(t, v, b)
    to = t[]==C_NULL ? PyObject(pynone) : pynewobject(t[])
    vo = v[]==C_NULL ? PyObject(pynone) : pynewobject(v[])
    bo = b[]==C_NULL ? PyObject(pynone) : pynewobject(b[])
    throw(PythonRuntimeError(to, vo, bo))
end

function pythrow(v::AbstractPyObject)
    throw(PythonRuntimeError(pytype(v), v, pynone))
end

function Base.showerror(io::IO, e::PythonRuntimeError)
    if pyisnone(e.t)
        print(io, "Python: mysterious error (no error was actually set)")
        return
    end

    # if this is a Julia exception then recursively print it and its stacktrace
    if pyerrmatches(e.t, pyjuliaexception)
        try
            jp = pyjuliavalue(e.v.args[0])
            if jp !== nothing
                je, jb = jp
                print(io, "Python: Julia: ")
                showerror(io, je)
                if jb === nothing
                    println(io)
                    print(io, "Stacktrace: none")
                else
                    io2 = IOBuffer()
                    Base.show_backtrace(IOContext(io2, :color=>true, :displaysize=>displaysize(io)), jb)
                    seekstart(io2)
                    printstyled(io, read(io2, String))
                end
            end
            if pyisnone(e.b)
                println(io)
                printstyled(io, "Python stacktrace: none")
                return
            else
                @goto pystacktrace
            end
        catch
            println(io, "<error while printing Julia excpetion inside Python exception>")
        end
    end

    # otherwise, print the Python exception
    print(io, "Python: ")
    try
        print(io, e.t.__name__)
    catch
        print(io, "<error while printing type>")
    end
    if !pyisnone(e.v)
        print(io, ": ")
        try
            print(io, e.v)
        catch
            print(io, "<error while printing value>")
        end
    end
    if !pyisnone(e.b)
        @label pystacktrace
        println(io)
        printstyled(io, "Python stacktrace:")
        try
            fs = pytracebackmodule.extract_tb(e.b)
            nfs = pylen(fs)
            for i in 1:nfs
                println(io)
                f = fs[nfs-i]
                printstyled(io, " [", i, "] ", )
                printstyled(io, f.name, bold=true)
                printstyled(io, " at ")
                printstyled(io, f.filename, ":", f.lineno, bold=true)
            end
        catch
            print(io, "<error while printing traceback>")
        end
    end
end
