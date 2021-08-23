errval(::C.PyPtr) = C.PyNULL
errval(::T) where {T<:Number} = zero(T) - one(T)

iserrval(val) = val == errval(val)

iserrset() = C.PyErr_Occurred() != C.PyNULL
iserrset(val) = val == errval(val)

errcheck() = iserrset() ? pythrow() : nothing
errcheck(val) = iserrset(val) ? pythrow() : val

iserrset_ambig(val) = iserrset(val) && iserrset()

errcheck_ambig(val) = iserrset_ambig(val) ? pythrow() : val

errclear() = C.PyErr_Clear()

errmatches(t) = (@autopy t C.PyErr_ExceptionMatches(getptr(t_))) == 1

function errget()
    t = Ref(C.PyNULL)
    v = Ref(C.PyNULL)
    b = Ref(C.PyNULL)
    C.PyErr_Fetch(t, v, b)
    (pynew(t[]), pynew(v[]), pynew(b[]))
end

errset(t::Py) = C.PyErr_SetNone(getptr(t))
errset(t::Py, v::Py) = C.PyErr_SetObject(getptr(t), getptr(v))
errset(t::Py, v::String) = C.PyErr_SetString(getptr(t), v)

function errnormalize!(t::Py, v::Py, b::Py)
    tptr = getptr(t)
    vptr = getptr(v)
    bptr = getptr(b)
    tref = Ref(tptr)
    vref = Ref(vptr)
    bref = Ref(bptr)
    C.PyErr_NormalizeException(tref, vref, bref)
    setptr!(t, tref[])
    setptr!(v, vref[])
    setptr!(b, bref[])
    (t, v, b)
end

mutable struct PyException <: Exception
    _t :: Py
    _v :: Py
    _b :: Py
    _isnormalized :: Bool
end

function Base.getproperty(exc::PyException, k::Symbol)
    if k in (:t, :v, :b) && !exc._isnormalized
        errnormalize!(exc._t, exc._v, exc._b)
        ispynull(exc._t) && setptr!(exc._t, incref(getptr(pybuiltins.None)))
        ispynull(exc._v) && setptr!(exc._v, incref(getptr(pybuiltins.None)))
        ispynull(exc._b) && setptr!(exc._b, incref(getptr(pybuiltins.None)))
        pyisnone(exc._v) || (exc._v.__traceback__ = exc._b)
        exc._isnormalized = true
    end
    k == :t ? exc._t : k == :v ? exc._v : k == :b ? exc._b : getfield(exc, k)
end

pythrow() = throw(PyException(errget()..., false))

file_to_pymodule(fname::String) = begin
    isfile(fname) || return nothing
    modules = pyimport("sys").modules
    for (n, m) in modules.items()
        if pyhasattr(m, "__file__")
            fname2 = pystr(String, m.__file__)
            if isfile(fname2) && realpath(fname) == realpath(fname2)
                return pystr(String, n)
            end
        end
    end
end

function Base.showerror(io::IO, e::PyException)
    print(io, "Python: ")

    if pyisnone(e.t)
        print(io, "mysterious error (no error was actually set)")
        return
    end

    if CONFIG.auto_sys_last_traceback
        try
            sys = pyimport("sys")
            sys.last_type = e.t
            sys.last_value = e.v
            sys.last_traceback = e.b
        catch err
            print(io, "<error while setting 'sys.last_traceback': $err>")
        end
    end

    # if this is a Julia exception then recursively print it and its stacktrace
    if !ispynull(pyJuliaError) && pyissubclass(e.t, pyJuliaError)
        try
            # Extract error value
            je, jb = pyconvert(Tuple{Any,Any}, e.v.args)
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
            if pyisnone(e.b)
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
        print(io, e.t.__name__)
    catch err
        print(io, "<error while printing type: $err>")
    end

    # print the error message
    if !pyisnone(e.v)
        print(io, ": ")
        try
            print(io, e.v)
        catch err
            print(io, "<error while printing value: $err>")
        end
    end

    # print the stacktrace
    if !pyisnone(e.b)
        @label pystacktrace
        println(io)
        printstyled(io, "Python stacktrace:")
        try
            fs = [(pystr(String, x.name), pystr(String, x.filename), pystr(String, x.lineno)) for x in pyimport("traceback").extract_tb(e.b)]
            if Base.VERSION < v"1.6.0-rc1"
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
                    if isfile(fname) && :stacktrace_contract_userdir in names(Base, all=true) && Base.stacktrace_contract_userdir()
                        if :replaceuserpath in names(Base, all=true)
                            fname = Base.replaceuserpath(fname)
                        elseif :contractuser in names(Base.Filesystem, all=true)
                            fname = Base.Filesystem.contractuser(fname)
                        end
                    end
                    printstyled(io, fname, ":", lineno, color = :light_black)
                end
            end
        catch err
            print(io, "<error while printing stacktrace: $err>")
        end
    end
end
