errcheck() = C.PyErr_Occurred() == C.PyNULL ? nothing : pythrow()
errcheck(ptr::C.PyPtr) = ptr == C.PyNULL ? pythrow() : ptr
errcheck(val::Number) = val == (zero(val) - one(val)) ? pythrow() : val
errcheck_nullable(ptr::C.PyPtr) = (ptr == C.PyNULL && errcheck(); ptr)

errclear() = C.PyErr_Clear()

function errget()
    t = Ref(C.PyNULL)
    v = Ref(C.PyNULL)
    b = Ref(C.PyNULL)
    C.PyErr_Fetch(t, v, b)
    (setptr!(pynew(), t[]), setptr!(pynew(), v[]), setptr!(pynew(), b[]))
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
    isnormalized :: Bool
end

function Base.getproperty(exc::PyException, k::Symbol)
    if k in (:t, :v, :b) && !exc.isnormalized
        errnormalize!(exc._t, exc._v, exc._b)
        ispynull(exc._t) && setptr!(exc._t, incref(getptr(pyNone)))
        ispynull(exc._v) && setptr!(exc._v, incref(getptr(pyNone)))
        ispynull(exc._b) && setptr!(exc._b, incref(getptr(pyNone)))
        exc.isnormalized = true
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

    if pyis(e.t, pyNone)
        print(io, "mysterious error (no error was actually set)")
        return
    end

    if CONFIG.sysautolasttraceback
        try
            sys = pyimport("sys")
            sys.last_type = e.t
            sys.last_value = e.v
            sys.last_traceback = e.b
        catch err
            print(io, "<error while setting 'sys.last_traceback': $err>")
        end
    end

    # # if this is a Julia exception then recursively print it and its stacktrace
    # if C.PyErr_GivenExceptionMatches(e.tref, C.PyExc_JuliaError()) != 0
    #     try
    #         # Extract error value
    #         vo = @pyv `$(e.vref).args[0]`::Any
    #         if vo isa Tuple{Exception,Any}
    #             je, jb = vo
    #         else
    #             je = vo
    #             jb = nothing
    #         end
    #         print(io, "Julia: ")
    #         # Show exception
    #         if je isa Exception
    #             showerror(io, je)
    #         else
    #             print(io, je)
    #         end
    #         # Show backtrace
    #         if jb === nothing
    #             println(io)
    #             printstyled(io, "Stacktrace: none")
    #         else
    #             io2 = IOBuffer()
    #             Base.show_backtrace(
    #                 IOContext(io2, io),
    #                 jb,
    #             )
    #             printstyled(io, String(take!(io2)))
    #         end
    #         # Show Python backtrace
    #         if isnull(e.bref)
    #             println(io)
    #             printstyled(io, "Python stacktrace: none")
    #             return
    #         else
    #             @goto pystacktrace
    #         end
    #     catch err
    #         println("<error while printing Julia exception inside Python exception: $err>")
    #     end
    # end

    # print the type name
    try
        print(io, e.t.__name__)
    catch err
        print(io, "<error while printing type: $err>")
    end

    # print the error message
    if !pyis(e.v, pyNone)
        print(io, ": ")
        try
            print(io, e.v)
        catch err
            print(io, "<error while printing value: $err>")
        end
    end

    # print the stacktrace
    if !pyis(e.b, pyNone)
        @label pystacktrace
        println(io)
        printstyled(io, "Python stacktrace:")
        try
            fs = [(pystr(String, x.name), pystr(String, x.filename), pystr(String, x.lineno)) for x in pyimport("traceback").extract_tb(e.b)]
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
