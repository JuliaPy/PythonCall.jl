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

errset(t::Py) = Base.GC.@preserve t C.PyErr_SetNone(getptr(t))
errset(t::Py, v::Py) = Base.GC.@preserve t v C.PyErr_SetObject(getptr(t), getptr(v))
errset(t::Py, v::String) = Base.GC.@preserve t C.PyErr_SetString(getptr(t), v)

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

"""
    PyException(x)

Wraps the Python exception `x` as a Julia `Exception`.
"""
mutable struct PyException <: Exception
    _t :: Py
    _v :: Py
    _b :: Py
    _isnormalized :: Bool
end
function PyException(v::Py=pybuiltins.None)
    if pyisnone(v)
        t = b = v
    elseif pyisinstance(v, pybuiltins.BaseException)
        t = pytype(v)
        b = pygetattr(v, "__traceback__", pybuiltins.None)
    else
        throw(ArgumentError("expecting a Python exception"))
    end
    PyException(t, v, b, true)
end
export PyException

ispy(x::PyException) = true
Py(x::PyException) = x.v

pyconvert_rule_exception(::Type{R}, x::Py) where {R<:PyException} = pyconvert_return(PyException(x))

function Base.show(io::IO, x::PyException)
    show(io, typeof(x))
    print(io, "(")
    show(io, x.v)
    print(io, ")")
end

function Base.getproperty(exc::PyException, k::Symbol)
    if k in (:t, :v, :b) && !exc._isnormalized
        errnormalize!(exc._t, exc._v, exc._b)
        pyisnull(exc._t) && setptr!(exc._t, incref(getptr(pybuiltins.None)))
        pyisnull(exc._v) && setptr!(exc._v, incref(getptr(pybuiltins.None)))
        pyisnull(exc._b) && setptr!(exc._b, incref(getptr(pybuiltins.None)))
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

Base.showerror(io::IO, e::PyException) = _showerror(io, e, nothing, backtrace=false)

Base.showerror(io::IO, e::PyException, bt; backtrace=true) = _showerror(io, e, bt; backtrace=backtrace)

function _showerror(io::IO, e::PyException, bt; backtrace=true)
    print(io, "Python: ")

    if pyisnone(e.t)
        print(io, "mysterious error (no error was actually set)")
        if backtrace
            Base.show_backtrace(io, bt)
        end
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

    if !pyisnull(pyJuliaError) && pyissubclass(e.t, pyJuliaError)
        # handle Julia exceptions specially
        try
            je, jb = pyconvert(Tuple{Any,Any}, e.v.args)
            print(io, "Julia: ")
            if je isa Exception
                showerror(io, je, jb, backtrace=backtrace && jb !== nothing)
            else
                print(io, je)
                backtrace && jb !== nothing && Base.show_backtrace(io, jb)
            end
        catch err
            println("<error while printing Julia exception inside Python exception: $err>")
        end
    else
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
    end

    if backtrace
        # print the Python stacktrace
        println(io)
        printstyled(io, "Python stacktrace:")
        if pyisnone(e.b)
            printstyled(io, " none")
        else
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

        # print the Julia stacktrace
        Base.show_backtrace(io, bt)
    end
end
