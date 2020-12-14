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
    end
    # make the code object
    co = CompiledCode.stash(newcode, "<julia $(src.file):$(src.line)>", mode)
    # go
    freelocals = locals === nothing ? :(GC.@preserve locals C.Py_DecRef(lptr)) : nothing
    quote
        # get the code pointer
        cptr = pyptr($co)
        # get the globals pointer
        globals = $(esc(:pyglobals))
        gptr = pyptr(globals)
        # ensure globals includes builtins
        if !globals.hasbuiltins
            if C.PyMapping_HasKeyString(gptr, "__builtins__") == 0
                err = C.PyMapping_SetItemString(gptr, "__builtins__", C.PyEval_GetBuiltins())
                ism1(err) && pythrow()
            end
            globals.hasbuiltins = true
        end
        # get locals (ALLOCATES lptr if locals===nothing)
        $(locals === nothing ? :(lptr = C.PyDict_New(); isnull(lptr) && pythrow()) : :(locals = $(esc(locals)); lptr = pyptr(locals)))
        # insert extra locals
        $([:(let; vo=C.PyObject_From($(esc(v))); isnull(vo) && ($freelocals; pythrow()); err=C.PyMapping_SetItemString(lptr, $(string(k)), vo); C.Py_DecRef(vo); ism1(err) && ($freelocals; pythrow()); end) for (k,v) in kvs]...)
        # Call eval (ALLOCATES rptr)
        rptr = GC.@preserve globals C.PyEval_EvalCode(cptr, gptr, lptr)
        isnull(rptr) && ($freelocals; pythrow())
        # extract values
        $(
            if mode == :eval
                quote
                    $freelocals
                    res = C.PyObject_As(rptr, $(esc(rettypearg)))
                    C.Py_DecRef(rptr)
                    res == PYERR() && pythrow()
                    res == NOTIMPLEMENTED() && (C.PyErr_SetString(C.PyExc_TypeError(), "Cannot convert return value of type '$(C.PyType_Name(C.Py_Type(rptr)))' to a Julia '$($(esc(rettypearg)))'"); pythrow())
                    return res::$(esc(rettypearg))
                end
            elseif mode == :exec
                quote
                    C.Py_DecRef(rptr)
                    $((((jv,jt) = (ex isa Expr && ex.head == :(::)) ? (ex.args[1], esc(ex.args[2])) : (ex, Any); quote
                        $(esc(jv)) = let
                            xo = C.PyMapping_GetItemString(lptr, $v)
                            isnull(xo) && ($freelocals; pythrow())
                            x = C.PyObject_As(xo, $jt)
                            x===NOTIMPLEMENTED() && C.PyErr_SetString(C.PyExc_TypeError(), "Cannot convert return value '$($(string(jv)))' of type '$(C.PyType_Name(C.Py_Type(xo)))' to a Julia '$($jt)'")
                            C.Py_DecRef(xo)
                            x===PYERR() && ($freelocals; pythrow())
                            x===NOTIMPLEMENTED() && ($freelocals; pythrow())
                            x::$jt;
                        end
                    end) for (ex,v,lhs) in zip(interps,intvars,islhs) if lhs)...)
                    $freelocals
                    nothing
                end
            else
                error()
            end
        )
    end
end

"""
    @py `...` [locals] [var=val, ...]

Executes the given Python code.

Julia values can be interpolated using the usual `\$(...)` syntax.
Additionally, assignment to interpolations is supported: e.g. `\$(x::T) = ...` will convert the right hand side to a `T` and assign it to `x`.

The globals are `pyglobals`. The locals are `locals`, if given, otherwise a temporary scope is created. Extra values to be interted into the scope can be given with extra `var=val` arguments.
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
