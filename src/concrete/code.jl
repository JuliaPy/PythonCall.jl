const MODULE_GLOBALS = Dict{Module,Py}()

function _pyeval_args(globals, locals)
    if globals isa Module
        globals_ = get!(pydict, MODULE_GLOBALS, globals)
    elseif ispy(globals)
        globals_ = globals
    else
        ArgumentError("globals must be a module or a Python dict")
    end
    if locals === nothing
        locals_ = Py(globals_)
    elseif ispy(locals)
        locals_ = Py(locals)
    else
        locals_ = pydict(locals)
    end
    return (globals_, locals_)
end

"""
    pyeval([T=Py], code, globals, locals=nothing)

Evaluate the given Python `code`, returning the result as a `T`.

If `globals` is a `Module`, then a persistent `dict` unique to that module is used.

If `locals` is not specified, then it is set to `globals`, i.e. the code runs in global scope.

For example the following computes `1.1+2.2` in the `Main` module as a `Float64`:
```
pyeval(Float64, "x+y", Main, (x=1.1, y=2.2))
```

See also [`@pyeval`](@ref).
"""
function pyeval(::Type{T}, code, globals, locals=nothing) where {T}
    globals_, locals_ = _pyeval_args(globals, locals)
    ans = pybuiltins.eval(code, globals_, locals_)
    pydel!(locals_)
    return T == Py ? ans : pyconvert_and_del(T, ans)
end
pyeval(code, globals, locals=nothing) = pyeval(Py, code, globals, locals)
export pyeval

_pyexec_ans(::Type{Nothing}, globals, locals) = nothing
@generated function _pyexec_ans(::Type{NamedTuple{names, types}}, globals, locals) where {names, types}
    # TODO: use precomputed interned strings
    # TODO: try to load from globals too
    n = length(names)
    code = []
    vars = Symbol[]
    for i in 1:n
        v = Symbol(:ans, i)
        push!(vars, v)
        push!(code, :($v = pyconvert_and_del($(types.parameters[i]), pygetitem(locals, $(string(names[i]))))))
    end
    push!(code, :(return $(NamedTuple{names, types})(($(vars...),))))
    return Expr(:block, code...)
end

"""
    pyexec([T=Nothing], code, globals, locals=nothing)

Execute the given Python `code`.

If `globals` is a `Module`, then a persistent `dict` unique to that module is used.

If `locals` is not specified, then it is set to `globals`, i.e. the code runs in global scope.

If `T==Nothing` then returns `nothing`. Otherwise `T` must be a concrete `NamedTuple` type
and the corresponding items from `locals` are extracted and returned.

For example the following computes `1.1+2.2` in the `Main` module as a `Float64`:
```
pyeval(@NamedTuple{ans::Float64}, "ans=x+y", Main, (x=1.1, y=2.2))
```

See also [`@pyexec`](@ref).
"""
function pyexec(::Type{T}, code, globals, locals=nothing) where {T}
    globals_, locals_ = _pyeval_args(globals, locals)
    pydel!(pybuiltins.exec(code, globals_, locals_))
    ans = _pyexec_ans(T, globals_, locals_)
    pydel!(locals_)
    return ans
end
pyexec(code, globals, locals=nothing) = pyexec(Nothing, code, globals, locals)
export pyexec

function _pyeval_macro_code(arg)
    if arg isa String
        return arg
    elseif arg isa Expr && arg.head === :macrocall && arg.args[1] == :(`foo`).args[1]
        return arg.args[3]
    else
        return nothing
    end
end

function _pyeval_macro_args(arg, filename, mode)
    # separate out inputs => code => outputs (with only code being required)
    if @capture(arg, inputs_ => code_ => outputs_)
        code = _pyeval_macro_code(code)
        code === nothing && error("invalid code")
    elseif @capture(arg, lhs_ => rhs_)
        code = _pyeval_macro_code(lhs)
        if code === nothing
            code = _pyeval_macro_code(rhs)
            code === nothing && error("invalid code")
            inputs = lhs
            outputs = nothing
        else
            inputs = nothing
            outputs = rhs
        end
    else
        code = _pyeval_macro_code(arg)
        code === nothing && error("invalid code")
        inputs = outputs = nothing
    end
    # precompile the code
    codestr = code
    codeobj = pynew()
    codeready = Ref(false)
    code = quote
        if !$codeready[]
            $pycopy!($codeobj, $pybuiltins.compile($codestr, $filename, $mode))
            $codeready[] = true
        end
        $codeobj
    end
    # convert inputs to locals
    if inputs === :GLOBAL
        locals = nothing
    elseif inputs === nothing
        locals = ()
    else
        if inputs isa Expr && inputs.head === :tuple
            inputs = inputs.args
        else
            inputs = [inputs]
        end
        locals = []
        for input in inputs
            if @capture(input, var_Symbol)
                push!(locals, var => var)
            elseif @capture(input, var_Symbol = ex_)
                push!(locals, var => ex)
            else
                error("invalid input: $input")
            end
        end
        locals = :(($([:($var = $ex) for (var,ex) in locals]...),))
    end
    # done
    return locals, code, outputs
end

"""
    @pyeval [inputs =>] code [=> T]

Evaluate the given `code` in a scope unique to the current module and return the answer as a `T`.

The `code` must be a literal string or command.

The `inputs` is a tuple of inputs of the form `v=expr` to be included in a temporary new
`locals` dict. Only `v` is required, `expr` defaults to `v`.

As a special case, if `inputs` is `GLOBAL` then the code is run in global scope.

For example the following computes `1.1+2.2` and returns a `Float64`:
```
@pyeval (x=1.1, y=2.2) => `x+y` => Float64
```
"""
macro pyeval(arg)
    locals, code, outputs = _pyeval_macro_args(arg, "$(__source__.file):$(__source__.line)", "eval")
    if outputs === nothing
        outputs = Py
    end
    esc(:($pyeval($outputs, $code, $__module__, $locals)))
end
export @pyeval

"""
    @pyexec [inputs =>] code [=> outputs]

Execute the given `code` in a scope unique to the current module.

The `code` must be a literal string or command.

The `inputs` is a tuple of inputs of the form `v=expr` to be included in a temporary new
`locals` dict. Only `v` is required, `expr` defaults to `v`.

As a special case, if `inputs` is `GLOBAL` then the code is run in global scope.

The `outputs` is a tuple of outputs of the form `x::T=v`, meaning that `v` is extracted from
locals, converted to `T` and assigned to `x`. Only `x` is required: `T` defaults to `Py`
and `v` defaults to `x`.

For example the following computes `1.1+2.2` and assigns its value to `ans` as a `Float64`:
```
@pyexec (x=1.1, y=2.2) => `ans=x+y` => ans::Float64
```
"""
macro pyexec(arg)
    locals, code, outputs = _pyeval_macro_args(arg, "$(__source__.file):$(__source__.line)", "exec")
    if outputs === nothing
        outputs = Nothing
        esc(:($pyexec(Nothing, $code, $__module__, $locals)))
    else
        if outputs isa Expr && outputs.head === :tuple
            oneoutput = false
            outputs = outputs.args
        else
            oneoutput = true
            outputs = [outputs]
        end
        pyvars = Symbol[]
        jlvars = Symbol[]
        types = []
        for output in outputs
            if @capture(output, lhs_ = rhs_)
                rhs isa Symbol || error("invalid output: $output")
                output = lhs
                pyvar = rhs
            else
                pyvar = missing
            end
            if @capture(output, lhs_ :: rhs_)
                outtype = rhs
                output = lhs
            else
                outtype = Py
            end
            output isa Symbol || error("invalid output: $output")
            if pyvar === missing
                pyvar = output
            end
            push!(pyvars, pyvar)
            push!(jlvars, output)
            push!(types, outtype)
        end
        outtype = :($NamedTuple{($(map(QuoteNode, pyvars)...),), Tuple{$(types...),}})
        ans = :($pyexec($outtype, $code, $__module__, $locals))
        if oneoutput
            ans = :($(jlvars[1]) = $ans[1])
        else
            if pyvars != jlvars
                outtype2 = :($NamedTuple{($(map(QuoteNode, jlvars)...),), Tuple{$(types...),}})
                ans = :($outtype2($ans))
            end
            ans = :(($(jlvars...),) = $ans)
        end
        esc(ans)
    end
end
export @pyexec
