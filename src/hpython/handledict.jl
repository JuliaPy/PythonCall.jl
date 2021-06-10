"""
    HandleDict{K}(py::Context, [dict])

A dictionary with keys of type `K` and values of type `PyHdl`.

It owns all the handles it has as values.
- `setindex!` steals the handle (if you don't want this then duplicate it first).
- `getindex`, `get`, `values`, `iterate` return borrowed handles (still owned by the dict).
- `empty!` and `delete!` close the handles of deleted items.
- `pop!` returns new handles (owned by the caller).

It has a finalizer which automatically closes all handles. You can also call `empty!` to
ensure any handles are closed immediately.

It is OK to store NULL handles.
"""
mutable struct HandleDict{K,D<:AbstractDict{K,PyHdl}} <: AbstractDict{K,PyHdl}
    ctx :: Context
    dict :: D
    function HandleDict{K,D}(ctx::Context, dict::D) where {K,D<:AbstractDict{K,PyHdl}}
        finalizer(new{K,D}(ctx, dict)) do x
            empty!(x)
        end
    end
end
HandleDict{K,D}(py::Context, dict) where {K, D<:AbstractDict{K,PyHdl}} = HandleDict{K,D}(py, convert(D, dict))
HandleDict{K,D}(py::Context) where {K, D<:AbstractDict{K,PyHdl}} = HandleDict{K,D}(py, D())
HandleDict{K}(py::Context, dict::D) where {K, D<:AbstractDict{K,PyHdl}} = HandleDict{K,D}(py, dict)
HandleDict{K}(py::Context, dict) where {K} = HandleDict{K}(py, convert(AbstractDict{K,PyHdl}, dict))
HandleDict{K}(py::Context) where {K} = HandleDict{K,Dict{K,PyHdl}}(py)
HandleDict(py::Context, dict::D) where {K, D<:AbstractDict{K}} = HandleDict{K}(py, dict)
HandleDict(py::Context) = HandleDict{Any}(py)
(b::Builtin{:HandleDict})(::Type{K}) where {K} = HandleDict{K}(b.ctx)

Base.length(x::HandleDict) = length(x.dict)

# returned handles are borrowed
Base.getindex(x::HandleDict, k) = getindex(x.dict, k)
Base.get(x::HandleDict, k, d) = get(x.dict, k, d)
Base.keys(x::HandleDict) = keys(x.dict)
Base.values(x::HandleDict) = values(x.dict)
Base.iterate(x::HandleDict) = iterate(x.dict)
Base.iterate(x::HandleDict, st) = iterate(x.dict, st)

# returned handles are now owned by caller
Base.pop!(x::HandleDict, k) = pop!(x.dict, k)
Base.pop!(x::HandleDict, k, d) = pop!(x.dict, k, d)

# closes the handle
function Base.delete!(x::HandleDict, k)
    h = pop!(x, k, PyNULL)
    h === PyNULL || x.ctx.closehdl(h)
    x
end

# steals the handle
function Base.setindex!(x::HandleDict, h::PyHdl, k)
    delete!(x, k)
    setindex!(x.dict, h, k)
    x
end

# closes all handles
function Base.empty!(x::HandleDict)
    for h in values(x.dict)
        h === PyNULL || x.ctx.closehdl(h)
    end
    empty!(x.dict)
    x
end
