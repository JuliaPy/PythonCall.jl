module Deps

import Conda, TOML, Pkg, Dates, ..PythonCall

### META

# increment whenever the format changes
const META_VERSION = 1

Base.@kwdef struct Meta
    timestamp::Float64
    load_path::Vector{String}
    version::VersionNumber
    files::Vector{String}
    conda_packages::Vector{String}
    conda_channels::Vector{String}
    pip_packages::Vector{String}
    pip_indexes::Vector{String}
    scripts::Vector{String}
end

function write_meta(io::IO, meta::Meta)
    write(io, Int(META_VERSION))
    write_meta(io, meta.timestamp)
    write_meta(io, meta.load_path)
    write_meta(io, meta.version)
    write_meta(io, meta.files)
    write_meta(io, meta.conda_packages)
    write_meta(io, meta.conda_channels)
    write_meta(io, meta.pip_packages)
    write_meta(io, meta.pip_indexes)
    write_meta(io, meta.scripts)
end

write_meta(io::IO, x::Float64) = write(io, x)
write_meta(io::IO, x::VersionNumber) = write_meta(io, string(x))
function write_meta(io::IO, x::String)
    write(io, Int(sizeof(x)))
    write(io, x)
end
function write_meta(io::IO, x::Vector)
    write(io, Int(length(x)))
    for y in x
        write_meta(io, y)
    end
end

function read_meta(io::IO)
    if read(io, Int) == META_VERSION
        Meta(
            timestamp = read_meta(io, Float64),
            load_path = read_meta(io, Vector{String}),
            version = read_meta(io, VersionNumber),
            files = read_meta(io, Vector{String}),
            conda_packages = read_meta(io, Vector{String}),
            conda_channels = read_meta(io, Vector{String}),
            pip_packages = read_meta(io, Vector{String}),
            pip_indexes = read_meta(io, Vector{String}),
            scripts = read_meta(io, Vector{String}),
        )
    end
end
read_meta(io::IO, ::Type{Float64}) = read(io, Float64)
read_meta(io::IO, ::Type{VersionNumber}) = VersionNumber(read_meta(io, String))
function read_meta(io::IO, ::Type{String})
    n = read(io, Int)
    bytes = read(io, n)
    length(bytes) == n || error()
    return String(bytes)
end
function read_meta(io::IO, ::Type{Vector{T}}) where {T}
    n = read(io, Int)
    x = Vector{T}()
    for i in 1:n
        push!(x, read_meta(io, T))
    end
    return x
end

const _meta_file = Ref("")

meta_file() = _meta_file[]

function load_meta()
    fn = meta_file()
    isfile(fn) ? open(read_meta, fn) : nothing
end

save_meta(meta::Meta) = open(io -> write_meta(io, meta), meta_file(), "w")

### CONDA

const _conda_env = Ref("")

conda_available() = !isempty(_conda_env[])

"""
    conda_env()

The path to the Conda environment in which Python dependencies are managed.
"""
function conda_env()
    ans = _conda_env[]
    isempty(ans) && error("Conda is not available")
    return ans
end

conda_script_dir() = Conda.script_dir(conda_env())

const CONDA_ENV_STACK = Vector{Dict{String,String}}()

function conda_activate()
    e = conda_env()
    push!(CONDA_ENV_STACK, copy(ENV))
    try
        # We run `conda shell.$shell activate $e` to get the shell script which activates
        # the environment, then parse its instructions (taking advantage of the fact that
        # conda outputs these instructions in quite a structured fashion) and make the
        # necessary modifications.
        #
        # Another approach is to use `conda shell.$shell+json activate $e` to get the
        # instructions as JSON, but as of July 2020 the latest version of conda on 32-bit
        # linux does not support +json.
        #
        # Another approach is to actually execute the activate script in a subshell and then
        # extract the resulting environment. This would capture environment variables set
        # in scripts, which are currently ignored, and so replicate the conda environment
        # precisely. Just need to work out how to get at the resulting environment.
        if Sys.iswindows()
            shell = "powershell"
            exportvarregex = r"^\$Env:([^ =]+) *= *\"(.*)\"$"
            setvarregex = exportvarregex
            unsetvarregex = r"^(Remove-Item +\$Env:/|Remove-Variable +)([^ =]+)$"
            runscriptregex = r"^\. +\"(.*)\"$"
        else
            shell = "posix"
            exportvarregex = r"^\\?export ([^ =]+)='(.*)'$"
            setvarregex = r"^([^ =]+)='(.*)'"
            unsetvarregex = r"^\\?unset +([^ ]+)$"
            runscriptregex = r"^\\?\. +\"(.*)\"$"
        end
        for line in eachline(Conda._set_conda_env(`$(Conda.conda) shell.$shell activate $e`))
            if (m = match(exportvarregex, line)) !== nothing
                @debug "conda activate export var" k=m.captures[1] v=m.captures[2]
                ENV[m.captures[1]] = m.captures[2]
            elseif (m = match(setvarregex, line)) !== nothing
                @debug "ignoring conda activate set var" k=m.captures[1] v=m.captures[2]
            elseif (m = match(unsetvarregex, line)) !== nothing
                @debug "conda activate unset var" k=m.captures[1]
                delete!(ENV, m.captures[1])
            elseif (m = match(runscriptregex, line)) !== nothing
                @debug "ignoring conda activate script" file=m.captures[1]
            elseif !isempty(strip(line))
                @warn "ignoring conda activate line" line
            end
        end
        @assert realpath(ENV["CONDA_PREFIX"]) == realpath(e)
        return
    catch
        conda_deactivate()
        rethrow()
    end
end

function conda_deactivate()
    env = pop!(CONDA_ENV_STACK)
    for k in collect(keys(ENV))
        delete!(ENV, k)
    end
    merge!(ENV, env)
    return
end

conda_run(cmd::Cmd) = Conda.runconda(cmd, conda_env())

conda_run_root(cmd::Cmd) = Conda.runconda(cmd)


### PYTHON

python_dir() = Conda.python_dir(conda_env())

python_exe() = joinpath(python_dir(), Sys.iswindows() ? "python.exe" : "python")

### PIP

pip_exe() = joinpath(conda_script_dir(), Sys.iswindows() ? "pip.exe" : "pip")

pip_enable() = Conda.pip_interop(true, conda_env())

function pip_run(cmd::Cmd)
    env = conda_env()
    @info("Running $(`pip $cmd`) in $(env) environment")
    run(Conda._set_conda_env(`$(pip_exe()) $cmd`, env))
    nothing
end

### RESOLVE

function can_skip_resolve()
    # resolve if the conda environment doesn't exist yet
    isdir(conda_env()) || return false
    # resolve if we haven't resolved before
    deps = load_meta()
    deps === nothing && return false
    # resolve whenever the PythonCall version changes
    deps.version == PythonCall.VERSION || return false
    # resolve whenever any of the environments in the load_path changes
    timestamp = max(deps.timestamp, stat(meta_file()).mtime)
    deps.load_path == Base.load_path() || return false
    for env in Base.load_path()
        proj = Base.env_project_file(env)
        dir = nothing
        if proj isa String
            dir = dirname(proj)
        elseif proj === true
            dir = env
        end
        dir === nothing && continue
        isdir(dir) || continue
        stat(dir).mtime < timestamp || return false
        fn = joinpath(dir, "PythonCallDeps.toml")
        if isfile(fn)
            stat(fn).mtime < timestamp || return false
        end
    end
    return true
end

"""
    user_deps_file()

The path to the `PythonCallDeps.toml` file in the active environment.
"""
function user_deps_file()
    for env in Base.load_path()
        proj = Base.env_project_file(env)
        if proj isa String
            return joinpath(dirname(proj), "PythonCallDeps.toml")
        elseif proj === true
            return joinpath(env, "PythonCallDeps.toml")
        end
    end
    error()
end

function deps_files()
    dirs = String[]
    for env in Base.load_path()
        proj = Base.env_project_file(env)
        if proj isa String
            push!(dirs, dirname(proj))
        elseif proj === true
            push!(dirs, env)
        end
    end
    for p in values(Pkg.dependencies())
        push!(dirs, p.source)
    end
    ans = String[]
    for dir in dirs
        fn = joinpath(dir, "PythonCallDeps.toml")
        if isfile(fn)
            push!(ans, fn)
        end
    end
    return ans
end

"""
    resolve(; create=true, force=false)

Resolve all Python dependencies.

If `create=true` then a new Conda environment is created and activated.
Otherwise, the existing one is updated.

By default, if no dependencies have actually changed, then resolving them is skipped.
Specify `force=true` to skip this check and force resolving dependencies.
"""
function resolve(; create=true, force=false)
    if !force && can_skip_resolve()
        # skip installing any dependencies
        create && conda_activate()

    else
        # find python dependencies
        all_deps_files = deps_files()
        conda_channels = String[]
        conda_packages = String[]
        pip_packages = String[]
        pip_indexes = String[]
        scripts = String[]
        for fn in all_deps_files
            @info "Found PythonCall dependencies at '$(fn)'"
            deps = TOML.parsefile(fn)
            if haskey(deps, "conda")
                if haskey(deps["conda"], "channels")
                    union!(conda_channels, deps["conda"]["channels"])
                end
                if haskey(deps["conda"], "packages")
                    union!(conda_packages, deps["conda"]["packages"])
                end
            end
            if haskey(deps, "pip")
                if haskey(deps["pip"], "indexes")
                    union!(pip_indexes, deps["pip"]["indexes"])
                end
                if haskey(deps["pip"], "packages")
                    union!(pip_packages, deps["pip"]["packages"])
                end
            end
            if haskey(deps, "script")
                if haskey(deps["script"], "expr")
                    push!(scripts, deps["script"]["expr"])
                end
                if haskey(deps["script"], "file")
                    push!(scripts, read(deps["script"]["file"], String))
                end
            end
        end

        # canonicalise dependencies
        sort!(unique!(conda_channels))
        sort!(unique!(conda_packages))
        sort!(unique!(pip_packages))
        sort!(unique!(pip_indexes))
        sort!(unique!(scripts))

        # determine if any dependencies have changed
        env = conda_env()
        skip = !force && isdir(env)
        if skip
            depinfo = load_meta()
            skip = (
                depinfo !== nothing &&
                conda_channels == depinfo.conda_channels &&
                conda_packages == depinfo.conda_packages &&
                pip_indexes == depinfo.pip_indexes &&
                pip_packages == depinfo.pip_packages &&
                scripts == depinfo.scripts
            )
        end

        # create and activate the conda environment with the desired packages
        # if update=true, just install the packages
        if skip
            # nothing has changed
            create && conda_activate()

        else
            conda_args = String[]
            for channel in conda_channels
                push!(conda_args, "--channel", channel)
            end
            append!(conda_args, conda_packages)
            if create || !isdir(env)
                ispath(env) && conda_run_root(`env remove --yes --prefix $env`)
                conda_run_root(`create --yes --no-default-packages --no-channel-priority --prefix $env $conda_args`)
                conda_activate()
            else
                conda_run(`install --yes --no-channel-priority --freeze-installed $conda_args`)
            end

            # install pip packages
            if !isempty(pip_packages)
                pip_enable()
                pip_args = String[]
                for index in pip_indexes
                    push!(pip_args, "--extra-index-url", index)
                end
                append!(pip_args, pip_packages)
                pip_run(`install $pip_args`)
            end

            # run scripts
            for script in scripts
                @info "Executing `$script`"
                eval(Meta.parse(script))
            end
        end

        # record what we did
        save_meta(Meta(
            timestamp = time(),
            load_path = Base.load_path(),
            version = PythonCall.VERSION,
            files = all_deps_files,
            conda_packages = conda_packages,
            conda_channels = conda_channels,
            pip_packages = pip_packages,
            pip_indexes = pip_indexes,
            scripts = scripts,
        ))
    end

    return
end

### INTERACTIVE

function spec_split(spec)
    spec = strip(spec)
    i = findfirst(c->isspace(c) || c in ('!','<','>','='), spec)
    if i === nothing
        return (spec, "")
    else
        return (spec[1:prevind(spec,i)], spec[i:end])
    end
end

"""
    status()

Display the status of dependencies of the current Julia project.
"""
function status()
    file = user_deps_file()
    printstyled("Status ", bold=true)
    println(file)
    if !isfile(file)
        println("(does not exist yet)")
        return
    end
    deps = TOML.parsefile(file)
    if haskey(deps, "conda")
        if haskey(deps["conda"], "channels") && !isempty(deps["conda"]["channels"])
            printstyled("Conda channels:", bold=true)
            println()
            for channel in deps["conda"]["channels"]
                println("  ", channel)
            end
        end
        if haskey(deps["conda"], "packages") && !isempty(deps["conda"]["packages"])
            printstyled("Conda packages:", bold=true)
            println()
            for spec in deps["conda"]["packages"]
                pkg, ver = spec_split(spec)
                print("  ", pkg, " ")
                printstyled(ver, color=:light_black)
                println()
            end
        end
    end
    if haskey(deps, "pip")
        if haskey(deps["pip"], "indexes") && !isempty(deps["pip"]["indexes"])
            printstyled("Pip indexes:", bold=true)
            println()
            for channel in deps["pip"]["indexes"]
                println("  ", channel)
            end
        end
        if haskey(deps["pip"], "packages") && !isempty(deps["pip"]["packages"])
            printstyled("Pip packages:", bold=true)
            println()
            for spec in deps["pip"]["packages"]
                pkg, ver = spec_split(spec)
                print("  ", pkg, " ")
                printstyled(ver, color=:light_black)
                println()
            end
        end
    end
    if haskey(deps, "script")
        if haskey(deps["script"], "expr")
            printstyled("Script expr:", bold=true)
            println()
            println("  ", deps["script"]["expr"])
        end
        if haskey(deps["script"], "file")
            printstyled("Script file:", bold=true)
            println()
            println("  ", deps["script"]["file"])
        end
    end
end

function getconvert!(::Type{T}, x, k) where {T}
    v = get!(T, x, k)
    if v isa T
        return v::T
    else
        v = convert(T, v)
        x[k] = v
        return x[k]::T
    end
end

"""
    add(...)

Add Python dependencies to the current Julia project.

Keyword arguments (all optional):
- `conda_channels`: An iterable of conda channels to use.
- `conda_packages`: An iterable of conda packages to install.
- `pip_indexes`: An iterable of pip indexes to use.
- `pip_packages`: An iterable of pip packages to install.
- `script_expr`: An expression to evaluate in the `Deps` module.
- `script_file`: The path to a Julia file to evaluate in the `Deps` module.
- `resolve=true`: When true, immediately resolve the dependencies. Otherwise, the
  dependencies are not resolved until you call [`resolve`](@ref) or load PythonCall in a
  new Julia session.
- `create=false`: When true, creates the environment from scratch when resolving.

The conda and pip packages can include version specifiers, such as `python>=3.6`.
"""
function add(; conda_channels=nothing, conda_packages=nothing, pip_indexes=nothing, pip_packages=nothing, script_expr=nothing, script_file=nothing, resolve=true, create=false)
    file = user_deps_file()
    deps = isfile(file) ? TOML.parsefile(file) : Dict{String,Any}()
    if conda_channels !== nothing
        sort!(union!(getconvert!(Vector{String}, getconvert!(Dict{String,Any}, deps, "conda"), "channels"), conda_channels))
    end
    if conda_packages !== nothing
        sort!(union!(getconvert!(Vector{String}, getconvert!(Dict{String,Any}, deps, "conda"), "packages"), conda_packages))
    end
    if pip_indexes !== nothing
        sort!(union!(getconvert!(Vector{String}, getconvert!(Dict{String,Any}, deps, "pip"), "indexes"), pip_indexes))
    end
    if pip_packages !== nothing
        sort!(union!(getconvert!(Vector{String}, getconvert!(Dict{String,Any}, deps, "pip"), "packages"), pip_packages))
    end
    if script_expr !== nothing
        getconvert!(Dict{String,Any}, deps, "script")["expr"] = script_expr
    end
    if script_file !== nothing
        getconvert!(Dict{String,Any}, deps, "script")["file"] = script_file
    end
    open(io->TOML.print(io, deps), file, "w")
    if resolve
        Deps.resolve(force=true, create=create)
    end
    return
end

"""
    rm(...)

Remove Python dependencies from the current Julia project.

Keyword arguments (all optional):
- `conda_channels`: An iterable of conda channels to remove.
- `conda_packages`: An iterable of conda packages to remove.
- `pip_indexes`: An iterable of pip indexes to remove.
- `pip_packages`: An iterable of pip packages to remove.
- `script_expr=false`: When true, remove the script expression.
- `script_file=false`: When true, remove the script file.
- `resolve=true`: When true, immediately resolve the dependencies. Otherwise, the
  dependencies are not resolved until you call [`resolve`](@ref) or load PythonCall in a
  new Julia session.
- `create=true`: When true, creates the environment from scratch when resolving.
"""
function rm(; conda_channels=nothing, conda_packages=nothing, pip_indexes=nothing, pip_packages=nothing, script_expr=false, script_file=false, resolve=true, create=true)
    file = user_deps_file()
    deps = isfile(file) ? TOML.parsefile(file) : Dict{String,Any}()
    if conda_channels !== nothing
        filter!(x -> x ∉ conda_channels, getconvert!(Vector{String}, getconvert!(Dict{String,Any}, deps, "conda"), "channels"))
    end
    if conda_packages !== nothing
        filter!(x -> spec_split(x)[1] ∉ conda_packages, getconvert!(Vector{String}, getconvert!(Dict{String,Any}, deps, "conda"), "packages"))
    end
    if pip_indexes !== nothing
        filter!(x -> x ∉ pip_indexes, getconvert!(Vector{String}, getconvert!(Dict{String,Any}, deps, "pip"), "indexes"))
    end
    if pip_packages !== nothing
        filter!(x -> spec_split(x)[1] ∉ pip_packages, getconvert!(Vector{String}, getconvert!(Dict{String,Any}, deps, "pip"), "packages"))
    end
    if script_expr
        delete!(getconvert!(Dict{String,Any}, deps, "script"), "expr")
    end
    if script_file !== nothing
        delete!(getconvert!(Dict{String,Any}, deps, "script"), "file")
    end
    open(io->TOML.print(io, deps), file, "w")
    if resolve
        Deps.resolve(force=true, create=create)
    end
    return
end

end
