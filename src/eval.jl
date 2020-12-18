pyeval_filename(src) = isfile(string(src.file)) ? "$(src.file):$(src.line)" : "julia:$(src.file):$(src.line)"

pyeval_macro(filename, mode, args...) = begin
    # find the code argument
    icode = findfirst(args) do x
        x isa Expr && x.head == :macrocall && x.args[1] == :(`foo`).args[1]
    end
    icode in (1,2) || error()
    code = args[icode].args[end]
    # the return type
    rettypearg = icode==2 ? args[1] : mode==:eval ? PyObject : Nothing
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
    islhs = [((match(r"[\n;]\s*$", codechunks[i])!==nothing) || (i==1 && match(r"^\s*$", codechunks[i])!==nothing)) && match(r"^\s*=($|[^=])", codechunks[i+1])!==nothing for (i,ex) in enumerate(interps)]
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
    if mode == :exec
        outkvts = [(ex isa Expr && ex.head == :(::)) ? (esc(ex.args[1]), v, esc(ex.args[2])) : (esc(ex), v, PyObject) for (ex,v,lhs) in zip(interps,intvars,islhs) if lhs]
    end
    # make the code object
    co = PyCode(newcode, filename, mode)
    # go
    freelocals = locals === nothing ? :(C.Py_DecRef(lptr)) : :(GC.@preserve locals nothing)
    ret = quote
        let
            # evaluate the inputs (so any errors are thrown before we have object references)
            $([:($(Symbol(:input,i)) = $(esc(v))) for (i,(k,v)) in enumerate(kvs)]...)
            # get the code pointer
            cptr = checknull(pyptr($co))
            # get the globals pointer
            globals = $(esc(:pyglobals))
            gptr = checknull(pyptr(globals))
            # ensure globals includes builtins
            if !globals.hasbuiltins
                if C.PyMapping_HasKeyString(gptr, "__builtins__") == 0
                    err = C.PyMapping_SetItemString(gptr, "__builtins__", C.PyEval_GetBuiltins())
                    ism1(err) && pythrow()
                end
                globals.hasbuiltins = true
            end
            # get locals (ALLOCATES lptr if locals===nothing)
            $(locals === nothing ? :(lptr = checknull(C.PyDict_New())) : :(locals = $(esc(locals)); lptr = checknull(pyptr(locals))))
            # insert extra locals
            $([:(let; vo=C.PyObject_From($(Symbol(:input,i))); isnull(vo) && ($freelocals; pythrow()); err=C.PyObject_SetItem(lptr, $(PyInternedString(string(k))), vo); C.Py_DecRef(vo); ism1(err) && ($freelocals; pythrow()); end) for (i,(k,v)) in enumerate(kvs)]...)
            # Call eval (ALLOCATES rptr)
            rptr = GC.@preserve globals C.PyEval_EvalCode(cptr, gptr, lptr)
            isnull(rptr) && ($freelocals; pythrow())
            # extract values
            $(
                if mode == :eval
                    quote
                        $freelocals
                        res = C.PyObject_Convert(rptr, $(esc(rettypearg)))
                        C.Py_DecRef(rptr)
                        ism1(res) && pythrow()
                        C.takeresult($(esc(rettypearg)))
                    end
                elseif mode == :exec
                    quote
                        C.Py_DecRef(rptr)
                        $((quote
                            $(Symbol(:output,i)) = let
                                xo = C.PyObject_GetItem(lptr, $(PyInternedString(v)))
                                isnull(xo) && ($freelocals; pythrow())
                                res = C.PyObject_Convert(xo, $t)
                                C.Py_DecRef(xo)
                                ism1(res) && ($freelocals; pythrow())
                                C.takeresult($t)
                            end
                        end for (i,(_,v,t)) in enumerate(outkvts))...)
                        $freelocals
                        ($((Symbol(:output,i) for i in 1:length(outkvts))...),)
                    end
                else
                    error()
                end
            )
        end
    end
    if mode == :exec
        ret = quote
            ($((k for (k,_,_) in outkvts)...),) = $ret
            nothing
        end
    end
    ret
end

"""
    @py `...` [locals] [var=val, ...]

Executes the given Python code.

Julia values can be interpolated using the usual `\$(...)` syntax.
Additionally, assignment to interpolations is supported: e.g. `\$(x::T) = ...` will convert the right hand side to a `T` and assign it to `x`.
Currently only single assignment is supported, and it must occur at the start of a line; multiple assignment (`\$x, \$y = ...`) or mutating assignment (`\$x += ...`) will not be recognized.

The globals are `pyglobals`. The locals are `locals`, if given, otherwise a temporary scope is created. Extra values to be interted into the scope can be given with extra `var=val` arguments.
"""
macro py(args...)
    pyeval_macro(pyeval_filename(__source__), :exec, args...)
end
export @py

"""
    py"..."

Shorthand for ```@py `...` ```.
"""
macro py_str(code::String)
    pyeval_macro(pyeval_filename(__source__), :exec, Expr(:macrocall, :(`foo`).args[1], code))
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
    pyeval_macro(pyeval_filename(__source__), :eval, args...)
end
export @pyv

"""
    pyv"..."

Shorthand for ```@pyv `...` ```.
"""
macro pyv_str(code::String)
    pyeval_macro(pyeval_filename(__source__), :eval, Expr(:macrocall, :(`foo`).args[1], code))
end
export @pyv_str

"""
    py`...` :: PyCode

A Python code object in "exec" mode which is compiled only once.

Suitable for using as the `code` argument to `pyeval`.
"""
macro py_cmd(code::String)
    PyCode(code, pyeval_filename(__source__), :exec)
end
export @py_cmd

"""
    pyv`...` :: PyCode

A Python code object in "eval" mode which is compiled only once.

Suitable for using as the `code` argument to `pyexec`.
"""
macro pyv_cmd(code::String)
    PyCode(code, pyeval_filename(__source__), :eval)
end
export @pyv_cmd
