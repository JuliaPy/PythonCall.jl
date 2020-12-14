function pyeval(::Type{R}, co::PyCode, globals::PyDict, locals::Union{PyDict,Nothing}=globals, extras::Union{NamedTuple,Nothing,Tuple{}}=nothing) where {R}
    # get code
    cptr = pyptr(co)
    # get globals & ensure __builtins__ is set
    gptr = pyptr(globals)
    if !globals.hasbuiltins
        if C.PyMapping_HasKeyString(gptr, "__builtins__") == 0
            err = C.PyMapping_SetItemString(gptr, "__builtins__", C.PyEval_GetBuiltins())
            ism1(err) && pythrow()
        end
        globals.hasbuiltins = true
    end
    # get locals (ALLOCATES lptr if locals===nothing)
    if locals === nothing
        lptr = C.PyDict_New()
        isnull(lptr) && pythrow()
    else
        lptr = pyptr(locals)
    end
    # insert extra locals
    if extras isa NamedTuple
        for (k,v) in pairs(extras)
            vo = C.PyObject_From(v)
            if isnull(vo)
                locals===nothing && C.Py_DecRef(lptr)
                pythrow()
            end
            err = C.PyMapping_SetItemString(lptr, string(k), vo)
            C.Py_DecRef(vo)
            if ism1(err)
                locals===nothing && C.Py_DecRef(lptr)
                pythrow()
            end
        end
    end
    # Call eval (ALLOCATES rptr)
    rptr = C.PyEval_EvalCode(cptr, gptr, lptr)
    if isnull(rptr)
        locals === nothing && C.Py_DecRef(lptr)
        pythrow()
    end
    # TODO: convert rptr using PyObject_As
    if co.mode == :exec
        if R <: Nothing
            C.Py_DecRef(rptr)
            locals===nothing && C.Py_DecRef(lptr)
            return nothing
        elseif R <: NamedTuple && isconcretetype(R)
            C.Py_DecRef(rptr)
            ret = C.PyMapping_ExtractAs(lptr, R)
            locals===nothing && C.Py_DecRef(lptr)
            ret === PYERR() && pythrow()
            ret === NOTIMPLEMENTED() && pythrow()
            return ret::R
        else
            C.Py_DecRef(rptr)
            locals===nothing && C.Py_DecRef(lptr)
            error("invalid return type $(R)")
        end
    elseif co.mode == :eval
        ret = C.PyObject_As(rptr, R)
        ret === NOTIMPLEMENTED() && C.PyErr_SetString(C.PyExc_TypeError(), "Cannot convert this '$(C.PyType_Name(C.Py_Type(rptr)))' to a Julia '$R'")
        C.Py_DecRef(rptr)
        locals===nothing && C.Py_DecRef(lptr)
        ret === PYERR() && pythrow()
        ret === NOTIMPLEMENTED() && pythrow()
        return ret::R
    else
        C.Py_DecRef(rptr)
        locals===nothing && C.Py_DecRef(lptr)
        error("invalid mode $(repr(co.mode))")
    end
end
export pyeval

module CompiledCode
    import ..Python: PyCode
    stash(co::PyCode) = begin
        nm = gensym()
        @eval $nm = $co
        co
    end
    stash(args...) = stash(PyCode(args...))
end

pyeval_macro(src, mode, args...) = begin
    # find the code argument
    icode = findfirst(args) do x
        x isa Expr && x.head == :macrocall && x.args[1] == :(`foo`).args[1]
    end
    icode in (1,2) || error()
    code = args[icode].args[end]
    # the return type
    rettypearg = icode==2 ? args[1] : mode==:eval ? Any : Nothing
    # parse the remaining arguments - the first one is optionally the locals, the rest are var=value pairs to include in the locals
    kvs = []
    locals = nothing
    for (i,arg) in enumerate(args[icode+1:end])
        if arg isa Expr && arg.head == :(=) && arg.args[1] isa Symbol
            push!(kvs, arg.args[1] => arg.args[2])
        elseif i == 1
            locals = arg
        else
            error()
        end
    end
    # find any interpolations
    i = firstindex(code)
    codechunks = String[]
    interps = []
    chunk = ""
    while true
        j = findnext('$', code, i)
        if j === nothing
            push!(codechunks, chunk * code[i:end])
            break
        elseif checkbounds(Bool, code, j+1) && code[j+1] == '$'
            chunk *= code[i:j]
            i = j+2
        else
            push!(codechunks, chunk * code[i:j-1])
            chunk = ""
            ex, i = Meta.parse(code, j+1, greedy=false)
            push!(interps, ex)
        end
    end
    # identify which are LHS and which are RHS
    # currently only the pattern "^ $(...) =" is recognized as a LHS
    # TODO: multiple assignment
    # TODO: mutating assignment
    islhs = [((match(r"\n\s*$", codechunks[i])!==nothing) || (i==1 && match(r"^\s*$", codechunks[i])!==nothing)) && match(r"\s*=", codechunks[i+1])!==nothing for (i,ex) in enumerate(interps)]
    # do the interpolation
    intvars = ["_jl_interp_$(i)_" for i in 1:length(interps)]
    for (k,v,lhs) in zip(intvars, interps, islhs)
        lhs || push!(kvs, Symbol(k) => v)
    end
    newcode = join([c * (checkbounds(Bool, intvars, i) ? islhs[i] ? intvars[i] : "($(intvars[i]))" : "") for (i,c) in enumerate(codechunks)])
    # for LHS interpolation, extract out type annotations
    if any(islhs)
        mode == :exec || error("interpolation on LHS only allowed in exec mode")
        icode == 1 || error("cannot specify return type when interpolation on LHS is used")
        rettypearg = :($NamedTuple{($([QuoteNode(Symbol(n)) for (n,lhs) in zip(intvars,islhs) if lhs]...),),Tuple{$([(ex isa Expr && ex.head == :(::)) ? ex.args[2] : Any for (ex,lhs) in zip(interps,islhs) if lhs]...)}})
    end
    # make the code object
    co = CompiledCode.stash(newcode, "<julia $(src.file):$(src.line)>", mode)
    # call pyeval
    ret = :(pyeval($(esc(rettypearg)), $(co), $(esc(:pyglobals)), $(esc(locals)), ($([:($k = $(esc(v))) for (k,v) in kvs]...),)))
    # assign
    if any(islhs)
        ret = :(($([(ex isa Expr && ex.head == :(::)) ? esc(ex.args[1]) : esc(ex) for (ex,lhs) in zip(interps,islhs) if lhs]...),) = $ret; nothing)
    end
    ret
end

"""
    @py [rettype] `...` [locals] [var=val, ...]

Executes the given Python code.

Julia values can be interpolated using the usual `\$(...)` syntax.
Additionally, assignment to interpolations is supported: e.g. `\$(x::T) = ...` will convert the right hand side to a `T` and assign it to `x`.

The globals are `pyglobals`. The locals are `locals`, if given, otherwise a temporary scope is created. Extra values to be interted into the scope can be given with extra `var=val` arguments.

The resulting expression has type `rettype`, which may be `Nothing` (the default) or `NamedTuple{names,types}` to extract variables with the given names as a named tuple.
"""
macro py(args...)
    pyeval_macro(__source__, :exec, args...)
end
export @py

"""
    py"..."

Shorthand for ```@py `...` ```.
"""
macro py_str(code::String)
    pyeval_macro(__source__, :exec, Expr(:macrocall, :(`foo`).args[1], code))
end
export @py_str

"""
    @pyv [rettype] `...` [locals] [var=val, ...]

Evaluate the given Python code.

Julia values can be interpolated using the usual `\$(...)` syntax.

The globals are `pyglobals`. The locals are `locals`, if given, otherwise a temporary scope is created. Extra values to be interted into the scope can be given with extra `var=val` arguments.

The result is converted to a `rettype`, which defaults to `PyObject`.
"""
macro pyv(args...)
    pyeval_macro(__source__, :eval, args...)
end
export @pyv

"""
    py`...` :: PyCode

A Python code object in "exec" mode which is compiled only once.

Suitable for using as the `code` argument to `pyeval`.
"""
macro py_cmd(code::String)
    CompiledCode.stash(code, "<julia $(__source__.file):$(__source__.line)>", :exec)
end
export @py_cmd

"""
    pyv`...` :: PyCode

A Python code object in "eval" mode which is compiled only once.

Suitable for using as the `code` argument to `pyexec`.
"""
macro pyv_cmd(code::String)
    CompiledCode.stash(code, "<julia $(__source__.file):$(__source__.line)>", :eval)
end
export @pyv_cmd
