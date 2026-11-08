# [Conversion to Julia](@id py2jl)

## [Conversion rules](@id py2jl-conversion)

Use `pyconvert(T, x)` to convert a Python object to a Julia type. With
`pyconvert(T, x, default)`, a declined conversion returns `default`.
Exceptions raised during conversion propagate in either form.
JuliaCall uses `T=Any` when passing Python arguments into Julia.

`Py` is the identity/fallback representation: `pyconvert(Py, x::Py)` returns
`x` directly. Rules should produce Julia representations rather than transform
one Python object into another through a `Py` target.

Canonical conversions prefer native Julia values: integers become `Int` or
`BigInt` according to their value; floats, complex numbers, strings and
`None` become `Float64`, `ComplexF64`, `String` and `nothing`.
Numeric conversion uses each supported source's representation: integral and
rational sources convert through Julia integers and rationals, while Python
`float` and `complex` use their stored floating-point representations. Supported
NumPy scalars have dedicated rules. Other numeric types need explicit rules;
membership in `numbers.Real` or `numbers.Complex`, or implementing `__float__`
or `__complex__`, does not by itself enable conversion. An exact conversion
that declines cannot be retried through those potentially inexact methods.

Lists become vectors, dictionaries become dictionaries, sets become sets,
and tuples become tuples (named tuples retain their names).
Array and buffer protocols produce native arrays before generic iterable
conversion. Explicit `PyList`, `PyDict`, `PySet` and `PyArray` targets
select Python-backed wrappers. Generic iterables remain lazy `PyIterable`
wrappers; canonical conversion never consumes an arbitrary iterator.

Native vector element types are inferred using Julia's `Base.promote_typejoin`,
without converting the individual values to a promoted numeric type.
Dictionary keys and values are inferred independently. Empty unconstrained
containers use `Any`. Explicit element types are honored; abstract bounds,
such as `Vector{<:Real}`, constrain each recursive conversion and the inferred
element type. Cyclic containers are unsupported and may overflow the stack.

## Preferences and context

```julia
pyconvert(Any, x; prefer=(
    "list" => PyList,
    "dict" => PyDict,
    "numpy:ndarray" => PyArray,
))
ctx = PyConvertContext(; prefer=("list" => PyList,))
pyconvert(ctx, Vector, x)
```

Preferences are tried in supplied order. Each matching source attempts normal
conversion to the intersection of the preferred type and the requested type.
A decline continues to the next preference; an exception propagates.
Ordinary conversion follows all preferences and does not reapply them at
that level. Preferences do not alter the registry or source ordering.
They are most useful for representation families; a narrow scalar preference
such as `Int8` can override a more appropriate canonical conversion.

Context propagation is explicit: a rule calls `pyconvert(ctx, T, child)`
to propagate preferences, or `pyconvert(T, child)` for independent conversion.
Built-in native vector and dictionary rules propagate context. A rule ignoring
context does not propagate preferences to its children.

## [Custom rules](@id py2jl-conversion-custom)

A rule advertises a Python source `t`, Julia output `T`, destination
`scope=S` (default `T`), and conversion function. Register rules in
`__init__`, rather than during precompilation.

```julia
function from_python(ctx, ::Type{MyArray}, x)
    child = pytryconvert(ctx, Vector, x)
    child === nothing && return nothing
    MyArray(something(child))
end
id = pyconvert_add_rule("builtins:list", MyArray, from_python; scope=MyArray)
pyconvert_replace_rule(id, "builtins:list", MyArray, from_python; scope=MyArray)
pyconvert_remove_rule(id)
```

Every add creates a unique, increasing ID, even for identical rules.
Replacement preserves the ID and creation position. Unknown IDs throw
`KeyError`. Additions, removals, replacements and resolution of pending
rules invalidate conversion caches.

A rule is eligible for requested type `R` when `typeintersect(T,R)`
is nonempty and `R <: S`. For normalized unions, at least one arm must
be a subtype of `S`. Julia's union normalization can erase an explicit
subtype opt-in: `Union{AbstractArray,MyArray}` is just `AbstractArray`.
Abstract and UnionAll output and scope types are supported.

Canonical rules use `scope=Any`. Only the owner of the Python source
should register them; PythonCall owns Python built-ins and standard-library
types for this purpose. Other packages must own their destination scope.
For example, a `list -> MyArray` rule scoped to `MyArray` cannot affect
`pyconvert(Any,x)` or `pyconvert(AbstractArray,x)`.

Rules return `nothing` to decline, an ordinary value to succeed, or
`Some(value)` for explicit success. Use `Some(nothing)` to successfully
return `nothing`, and `Some(Some(value))` to return an actual `Some`.
Prefer returning ordinary values directly. An unnecessary `Some` can allocate
when its concrete type crosses a dispatch boundary; built-in rules wrap only
successful `nothing` and `Some` values.
`pytryconvert([ctx,] T,x)` always returns `nothing` on failure and
`Some(value)` on success. Both `f(ctx,T,x)` and `f(T,x)` rules are
supported; the context-aware form is preferred when applicable to that target.
A rule receives a non-union target: each arm of the output/request
intersection is tried before moving to the next rule. Union arm order is
unspecified. Declines continue through all eligible rules. Exceptions propagate.
There is an implicit final `Py` fallback only when the requested type admits it.

### Source ordering

Registered source types form a global DAG using strict Python `issubclass`
relationships, including ABC relationships. More-specific sources precede
their ancestors. Ordering never depends on the converted object's MRO or on
Julia output-type specificity. Per-call rule selection filters this global
order without reordering it.

Independent virtual matchers `<arraystruct>`, `<arrayinterface>`,
`<array>` and `<buffer>` test the corresponding type attributes or buffer
support. A supporting registered real type precedes the protocol; protocols
precede `object`. No array protocol falsely implies buffer conformance.

Internal source priorities put Julia wrappers first, then array protocols,
then buffers, then ordinary types. Subtypes inherit the maximum ancestor
priority. Within a priority, stable topological sorting respects subtype
edges and breaks ties by source registration order. Rules on the same source
run in creation order. Incomparable array matchers use definition order:
array struct, array interface, then array. A specific ndarray rule can
override its generic protocol rules.

### Lazy resolution and diagnostics

Source names in rules and preferences use `"module:name"`, where `name` is an
attribute path (possibly dotted) looked up on the loaded module. It need not
match the type's `__module__` or `__qualname__`: a public re-export such as
`"mypackage:PublicType"` works even if the type is defined in an internal module.
A bare name abbreviates a built-in: `"list"` means `"builtins:list"`, so
`prefer=("list" => PyList,)` opts into list wrappers. Aliases resolving to the
same Python type share source ordering; their rules retain creation order.

Optional modules are never imported merely to resolve rules or preferences.
Pending rules enter the global order when their module is
available in `sys.modules` and the named type can be resolved. Encountering
a new Python source module checks pending rules; cache hits do not.
Modules should define their relevant types and complete ABC registrations
before conversion begins. Arbitrary later `ABC.register()` mutations are
unsupported. Cached Python type identities retain owning references.

`pyconvert_rules(T,x)` lists source-matching rules in global order, including
ID, source, advertised output, scope, effective source priority and destination
eligibility. The internal `PythonCall.Convert.pyconvert_preferred_type(t)`
reports the first canonical candidate, which may decline for a particular value.

```@docs
PythonCall.pyconvert_add_rule
PythonCall.PyConvertContext
PythonCall.pytryconvert
PythonCall.pyconvert_remove_rule
PythonCall.pyconvert_replace_rule
PythonCall.pyconvert_rules
```
