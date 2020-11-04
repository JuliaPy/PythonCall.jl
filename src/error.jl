cpyerrtype() = cpycall_raw(Val(:PyErr_Occurred), CPyPtr)

pyerroccurred() = cpyerrtype() != C_NULL
function pyerroccurred(t::AbstractPyObject)
    e = cpyerrtype()
    e != C_NULL && cpycall_boolx(Val(:PyErr_GivenExceptionMatches), e, t)
end

pyerrclear() = cpycall_raw(Val(:PyErr_Clear), Cvoid)

pyerrcheck() = pyerroccurred() ? pythrow() : nothing

pyerrset(t::AbstractPyObject) = cpycall_voidx(Val(:PyErr_SetNone), t)
pyerrset(t::AbstractPyObject, x::AbstractString) = cpycall_voidx(Val(:PyErr_SetString), t, x)
pyerrset(t::AbstractPyObject, x::AbstractPyObject) = cpycall_voidx(Val(:PyErr_SetObject), t, x)

struct PythonRuntimeError <: Exception
    t :: PyObject
    v :: PyObject
    b :: PyObject
end

function pythrow()
    t = Ref{CPyPtr}()
    v = Ref{CPyPtr}()
    b = Ref{CPyPtr}()
    cpycall_raw(Val(:PyErr_Fetch), Cvoid, t, v, b)
    cpycall_raw(Val(:PyErr_NormalizeException), Cvoid, t, v, b)
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

    # # if this is a Julia exception then recursively print it and its stacktrace
    # if pyerror_givenexceptionmatches(e.t, pyexc_JuliaException_type())
    #     # get the exception and backtrace
    #     jp = try
    #         pyjulia_getvalue(e.v.args[0])
    #     catch
    #         @goto jlerr
    #     end
    #     if jp isa Exception
    #         je = jp
    #         jb = nothing
    #     elseif jp isa Tuple{Exception, AbstractVector}
    #         je, jb = jp
    #     else
    #         @goto jlerr
    #     end

    #     showerror(io, je)
    #     if jb === nothing
    #         println(io)
    #         print(io, "<no stacktrace>")
    #     else
    #         io2 = IOBuffer()
    #         Base.show_backtrace(IOContext(io2, :color=>true, :displaysize=>displaysize(io)), jb)
    #         seekstart(io2)
    #         printstyled(io, read(io2, String))
    #     end

    #     if !isnull(e.b)
    #         @goto pystacktrace
    #     else
    #         println(io)
    #         printstyled(io, "(thrown from unknown Python code)")
    #         return
    #     end

    #     @label jlerr
    #     println(io, "<error while printing Julia exception inside Python exception>")
    # end

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
            fs = pyimport("traceback").extract_tb(e.b)
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
