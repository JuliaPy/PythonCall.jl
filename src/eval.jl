const SCOPES = Dict{Any,PyObject}()
const COMPILECACHE = Dict{Tuple{String,String,String},PyObject}()

scope(s) = get!(pydict, SCOPES, s)
scope(s::AbstractPyObject) = s

"""
    pyeval(src, scope, locals=nothing)

Evaluate Python expression `src` in the context of the given `scope` and `locals`. Return the value.

If the `scope` is a Python object, it must be a `dict` to use as globals.

Otherwise, a globals dict is created and reused for each unique `scope`. For example `pyeval(src, @__MODULE__)` evaluates in a scope unique to the current module.
"""
pyeval(src, globals, locals=nothing) = pyevalfunc(src, scope(globals), locals)

"""
    pyexec(src, scope, locals=nothing)

Execute Python expression `src` in the context of the given `scope` and `locals`.

If the `scope` is a Python object, it must be a `dict` to use as globals.

Otherwise, a globals dict is created and reused for each unique `scope`. For example `pyexec(src, @__MODULE__)` executes in a scope unique to the current module.
"""
pyexec(src, globals, locals=nothing) = (pyexecfunc(src, scope(globals), locals); nothing)

"""
    py"...."[flags]

Evaluate (`v`) or execute (`x`) the given Python source code.

Julia values may be interpolated into the source code with `\$` syntax. For a literal `\$`, enter `\$\$`.

Execution occurs in a global scope unique to the current module.

The flags can be any combination of the following characters:
- `v` (evaluate): Evaluate a single expression and return its value.
- `x` (execute): Execute the code and return `nothing`.
- `g` (globals): Return the dict of globals for the scope instead.
- `l` (locals): Perform the computation in a new local scope and return that scope.
- `c` (compile): Cache a compiled version of the source and re-use it each time, for speed.

If neither `v` nor `x` is specified and the code is a single line then `v` (evaluate) is assumed, otherwise `x` (execute).
"""
macro py_str(src::String, flags::String="")
    # parse the flags
    exec = '\n' in src
    retglobals = false
    retlocals = false
    lazy = false
    compile = false
    for f in flags
        if f == 'x'
            exec = true
        elseif f == 'v'
            exec = false
        elseif f == 'g'
            retglobals = true
        elseif f == 'l'
            retlocals = true
        # elseif f == 'z'
        #     lazy = true
        elseif f == 'c'
            compile = true
        else
            error("invalid flags: `py\"...\"$flags`")
        end
    end
    # parse src for $-interpolations
    chunks = String[]
    interps = []
    i = firstindex(src)
    while true
        j = findnext('$', src, i)
        if j === nothing
            push!(chunks, src[i:end])
            break
        else
            push!(chunks, src[i:prevind(src,j)])
        end
        if checkbounds(Bool, src, j+1) && src[j+1] == '$'
            push!(chunks, "\$")
            i = j+2
        else
            ex, i = Meta.parse(src, j+1, greedy=false)
            var = "_jl_interp_$(length(interps)+1)_"
            push!(interps, var => ex)
            push!(chunks, "($var)")
        end
    end
    newsrcstr = newsrc = join(chunks)
    # compile the code so there is string information
    cfile = "julia:$(__source__.file):$(__source__.line)"
    cmode = exec ? "exec" : "eval"
    newsrc = :(pycompile($newsrcstr, $cfile, $cmode))
    if compile
        # compile the code lazily (so the python parser is only invoked once)
        # Julia crashes if you try to put PyLazyObject(()->pycompile(...)) directly in the syntax tree. Is this a bug??
        # Instead, we use a run-time lookup table.
        newsrc = :(get!(()->$newsrc, COMPILECACHE, ($newsrcstr, $cfile, $cmode)))
    end
    # make the expression
    ex = :(let
        globals = scope(@__MODULE__)
        locals = $(retlocals ? :(pydict()) : :globals)
        $([:(check(C.PyDict_SetItemString(globals, $k, pyobject($(esc(v)))))) for (k,v) in interps]...)
        result = $(exec ? :pyexec : :pyeval)($newsrc, globals, locals)
        $([:(check(C.PyDict_DelItemString(globals, $k))) for (k,v) in interps]...)
        $(retlocals ? :locals : retglobals ? :globals : exec ? nothing : :result)
    end)
    if lazy
        # wrap as a lazy object
        ex = :(PyLazyObject(() -> $ex))
    end
    ex
end
export @py_str
