pyeval_filename(src) =
    isfile(string(src.file)) ? "$(src.file):$(src.line)" : "julia:$(src.file):$(src.line)"

pyeval_macro(filename, mode, codearg, args...) = begin
    # unwrap rettype
    if codearg isa Expr && codearg.head == :(::)
        rettypearg = codearg.args[end]
        codearg = codearg.args[1]
    else
        rettypearg = mode == :exec ? Nothing : PyObject
    end
    # extract the code
    code = codearg.args[end]
    # parse the remaining arguments - the first one is optionally the locals, the rest are var=value pairs to include in the locals
    kvs = []
    locals = nothing
    for (i, arg) in enumerate(args)
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
        j = findnext(==('$'), code, i)
        if j === nothing
            push!(codechunks, chunk * code[i:end])
            break
        elseif checkbounds(Bool, code, j + 1) && code[j+1] == '$'
            chunk *= code[i:j]
            i = j + 2
        else
            push!(codechunks, chunk * code[i:j-1])
            chunk = ""
            ex, i = Meta.parse(code, j + 1, greedy = false)
            push!(interps, ex)
        end
    end
    # identify which are LHS and which are RHS
    # currently only the pattern "^ $(...) =" is recognized as a LHS
    # TODO: multiple assignment
    # TODO: mutating assignment
    islhs = [
        (
            (match(r"[\n;=]\s*$", codechunks[i]) !== nothing) ||
            (i == 1 && match(r"^\s*$", codechunks[i]) !== nothing)
        ) && match(r"^\s*=($|[^=])", codechunks[i+1]) !== nothing for
        (i, ex) in enumerate(interps)
    ]
    # do the interpolation
    intvars = ["_jl_interp_$(i)_" for i = 1:length(interps)]
    for (k, v, lhs) in zip(intvars, interps, islhs)
        lhs || push!(kvs, Symbol(k) => v)
    end
    newcode = join([
        c * (
            checkbounds(Bool, intvars, i) ? islhs[i] ? intvars[i] : "($(intvars[i]))" : ""
        ) for (i, c) in enumerate(codechunks)
    ])
    # for LHS interpolation, extract out type annotations
    if any(islhs)
        mode == :exec || error("interpolation on LHS only allowed in exec mode")
    end
    if mode == :exec
        outkvts = [
            (ex isa Expr && ex.head == :(::)) ?
            (esc(ex.args[1]), v, esc(ex.args[2])) : (esc(ex), v, PyObject) for
            (ex, v, lhs) in zip(interps, intvars, islhs) if lhs
        ]
    end
    # make the code be a function body
    if mode == :execr
        newcode = "def _jl_tmp_ans_func_($(join(intvars, ", "))):\n" * join(map(line->"    "*line, split(newcode, "\n")), "\n") * "\n\nans = _jl_tmp_ans_func_($(join(intvars, ", ")))"
    end
    # make the code object
    co = PyCode(newcode, filename, mode == :eval ? :eval : :exec)
    # go
    freelocals =
        locals === nothing ? :(C.Py_DecRef(lptr)) : :(GC.@preserve locals nothing)
    ret = quote
        let
            # evaluate the inputs (so any errors are thrown before we have object references)
            $([:($(Symbol(:input, i)) = $(esc(v))) for (i, (k, v)) in enumerate(kvs)]...)
            # get the code pointer
            cptr = checknull(pyptr($co))
            # get the globals pointer
            globals = $(esc(:pyglobals))
            gptr = checknull(pyptr(globals))
            # ensure globals includes builtins
            if globals isa PyDict && !globals.hasbuiltins
                if C.PyMapping_HasKeyString(gptr, "__builtins__") == 0
                    err = C.PyMapping_SetItemString(
                        gptr,
                        "__builtins__",
                        C.PyEval_GetBuiltins(),
                    )
                    ism1(err) && pythrow()
                end
                globals.hasbuiltins = true
            end
            # get locals (ALLOCATES lptr if locals===nothing)
            $(
                locals === nothing ? :(lptr = checknull(C.PyDict_New())) :
                :(locals = $(esc(locals)); lptr = checknull(pyptr(locals)))
            )
            # insert extra locals
            $(
                [
                    :(
                        let
                            vo = C.PyObject_From($(Symbol(:input, i)))
                            isnull(vo) && ($freelocals; pythrow())
                            err = C.PyObject_SetItem(
                                lptr,
                                $(PyInternedString(string(k))),
                                vo,
                            )
                            C.Py_DecRef(vo)
                            ism1(err) && ($freelocals; pythrow())
                        end
                    ) for (i, (k, v)) in enumerate(kvs)
                ]...
            )
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
                elseif mode in (:execa, :execr)
                    quote
                        C.Py_DecRef(rptr)
                        xo = C.PyObject_GetItem(lptr, $(PyInternedString("ans")))
                        isnull(xo) && ($freelocals; pythrow())
                        res = C.PyObject_Convert(xo, $(esc(rettypearg)))
                        C.Py_DecRef(xo)
                        $freelocals
                        ism1(res) && pythrow()
                        C.takeresult($(esc(rettypearg)))
                    end
                elseif mode == :exec
                    quote
                        C.Py_DecRef(rptr)
                        $(
                            (
                                quote
                                    $(Symbol(:output, i)) =
                                        let
                                            xo = C.PyObject_GetItem(
                                                lptr,
                                                $(PyInternedString(v)),
                                            )
                                            isnull(xo) && ($freelocals; pythrow())
                                            res = C.PyObject_Convert(xo, $t)
                                            C.Py_DecRef(xo)
                                            ism1(res) && ($freelocals; pythrow())
                                            C.takeresult($t)
                                        end
                                end for (i, (_, v, t)) in enumerate(outkvts)
                            )...
                        )
                        $freelocals
                        ($((Symbol(:output, i) for i = 1:length(outkvts))...),)
                    end
                else
                    error()
                end
            )
        end
    end
    if mode == :exec
        ret = quote
            ($((k for (k, _, _) in outkvts)...),) = $ret
            nothing
        end
    end
    ret
end

"""
    @py `...` [locals] [var=val, ...]

Execute the given Python code.

Julia values can be interpolated using the usual `\$(...)` syntax.

Additionally, assignment to interpolations is supported: e.g. `\$(x::T) = ...` will convert the right hand side to a `T` and assign it to `x`.
- Currently only single assignment is supported. Multiple assignment (`\$x, \$y = ...`) or mutating assignment (`\$x += ...`) will not be recognized.
- What actually happens is that the assignment is to a temporary Python variable, which is then read when execution successfully finishes.
  Hence if an exception occurs, no assignments will happen.

The globals are `pyglobals`.
The locals are `locals`, if given, otherwise a temporary scope is created. Extra values to be interted into the scope can be given with extra `var=val` arguments.
"""
macro py(args...)
    pyeval_macro(pyeval_filename(__source__), :exec, args...)
end
export @py

"""
    @pyg `...` [var=val, ...]

Execute the given Python code in the global scope.

This is simply shorthand for ```@py `...` pyglobals ``` (see [`@py`](@ref)).
"""
macro pyg(code, args...)
    pyeval_macro(pyeval_filename(__source__), :exec, code, :pyglobals, args...)
end
export @pyg

"""
    @pyv `...`[::rettype] [locals] [var=val, ...]

Evaluate the given Python expression and return its value.

Julia values can be interpolated using the usual `\$(...)` syntax.

The globals are `pyglobals`.
The locals are `locals`, if given, otherwise a temporary scope is created. Extra values to be interted into the scope can be given with extra `var=val` arguments.

The result is converted to a `rettype`, which defaults to `PyObject`.
"""
macro pyv(args...)
    pyeval_macro(pyeval_filename(__source__), :eval, args...)
end
export @pyv

"""
    @pya `...`[::rettype] [locals] [var=val, ...]

Execute the given Python code and return `ans`.

This is the same as `@py ...` except that the variable `ans` is extracted from the scope and returned.
"""
macro pya(args...)
    pyeval_macro(pyeval_filename(__source__), :execa, args...)
end
export @pya

"""
    @pyr `...`[::rettype] [locals] [var=val, ...]

Execute the given Python code in a function and return its return value.

Essentially equivalent to ```@pya `def result(): ...; ans = result()` ```.
"""
macro pyr(args...)
    pyeval_macro(pyeval_filename(__source__), :execr, args...)
end
export @pyr

"""
    py`...` :: PyCode

Literal syntax for a compiled [`PyCode`](@ref) object in "exec" mode.

Suitable for passing to Python's `exec` function.
"""
macro py_cmd(code::String)
    PyCode(code, pyeval_filename(__source__), :exec)
end
export @py_cmd

"""
    pyv`...` :: PyCode

Literal syntax for a compiled [`PyCode`](@ref) object in "eval" mode.

Suitable for passing to Python's `eval` function.
"""
macro pyv_cmd(code::String)
    PyCode(code, pyeval_filename(__source__), :eval)
end
export @pyv_cmd
