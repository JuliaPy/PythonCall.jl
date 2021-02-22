# The Python module `julia`

For interactive or scripting use, the simplest way to get started is:

```python
from julia import Main as jl
```

This loads a single variable `jl` (a [`ModuleValue`](#julia.ModuleValue)) which represents the `Main` module in Julia, from which all of Julia's functionality is available.

If you are writing a package which uses Julia, then to avoid polluting the global `Main` namespace you should do:

```python
import julia; jl = julia.newmodule("SomeName");
```

Now you can do `jl.rand(jl.Bool, 5, 5)`, which is equivalent to `rand(Bool, 5, 5)` in Julia.

When a Python value is passed to Julia, then typically it will be converted according to [this table](../conversion/#Python-to-Julia) with `T=Any`. Sometimes a more specific type will be used, such as when assigning to an array whose element type is known.

When a Julia value is returned to Python, it will normally be converted according to [this table](../conversion/#Julia-to-Python).

## Wrapper types

Apart from a few fundamental immutable types (see [here](../conversion/#Julia-to-Python)), all Julia values are by default converted into Python to some [`AnyValue`](#julia.AnyValue) object, which wraps the original value. Some types are converted to a subclass of [`AnyValue`](#julia.AnyValue) which provides additional Python semantics --- e.g. Julia vectors are interpreted as Python sequences.

There is also a [`RawValue`](#julia.RawValue) object, which gives a stricter "Julia-only" interface, documented below. These types all inherit from `ValueBase`:

- `ValueBase`
  - [`RawValue`](#julia.RawValue)
  - [`AnyValue`](#julia.AnyValue)
    - [`NumberValue`](#julia.NumberValue)
      - `ComplexValue`
      - `RealValue`
        - `RationalValue`
        - `IntegerValue`
    - [`ArrayValue`](#julia.ArrayValue)
      - `VectorValue`
    - [`DictValue`](#julia.DictValue)
    - [`SetValue`](#julia.SetValue)
    - [`IOValue`](#julia.IOValue)
      - `RawIOValue`
      - `BufferedIOValue`
      - `TextIOValue`
    - [`ModuleValue`](#julia.ModuleValue)
    - [`TypeValue`](#julia.TypeValue)

```@raw html
<article class="docstring">
    <header>
        <a class="docstring-binding" id="julia.AnyValue" href="#julia.AnyValue">julia.AnyValue</a>
        —
        <span class="docstring-category">Python Class</span>
    </header>
    <section>
        Wraps any Julia object, giving it some basic Python semantics. Subtypes provide extra semantics.

        Supports `repr(x)`, `str(x)`, attributes (`x.attr`), calling (`x(a,b)`), iteration, comparisons, `len(x)`, `a in x`, `dir(x)`.

        Calling, indexing, attribute access, etc. will convert the result to a Python object according to <a href="../conversion/#Julia-to-Python">this table</a>.
        This is typically a builtin Python type (for immutables) or a subtype of `AnyValue`.

        Attribute access can be used to access Julia properties as well as normal class members.
        In the case of a name clash, the class member will take precedence.
        For convenience with Julia naming conventions, `_b` at the end of an attribute is replaced with `!` and `_bb` is replaced with `!!`.

        <h6>Members</h6>
        <ul>
            <li><code>_jl_raw()</code>: Convert to a <a href="#julia.RawValue"><code>RawValue</code></a></li>
            <li><code>_jl_display()</code>: Display the object using Julia's display mechanism.</li>
            <li><code>_jl_help()</code>: Display help for the object.</li>
        </ul>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="julia.NumberValue" href="#julia.NumberValue">julia.NumberValue</a>
        —
        <span class="docstring-category">Python Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>Number</code> value. It is a subclass of <code>numbers.Number</code> and behaves similar to other Python numbers.</p>
        <p>There are also subtypes <code>ComplexValue</code>, <code>RealValue</code>, <code>RationalValue</code>, <code>IntegerValue</code> which wrap values of the corresponding Julia types, and are subclasses of the corresponding <code>numbers</code> ABC.</p>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="julia.ArrayValue" href="#julia.ArrayValue">julia.ArrayValue</a>
        —
        <span class="docstring-category">Python Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>AbstractArray</code> value. It is a subclass of <code>collections.abc.Collection</code>.</p>
        <p>It supports zero-up indexing, and can be indexed with integers or slices. Slicing returns a view of the original array.</p>
        <p>There is also the subtype <code>VectorValue</code> which wraps any <code>AbstractVector</code>. It is a subclass of <code>collections.abc.Sequence</code> and behaves similar to a Python <code>list</code>.</p>
        <p>If the array is strided and its eltype is supported (i.e. <code>Bool</code>, <code>IntXX</code>, <code>UIntXX</code>, <code>FloatXX</code>, <code>Complex{FloatXX}</code>, <code>Ptr{Cvoid}</code> or <code>Tuple</code> or <code>NamedTuple</code> of these) then it supports the buffer protocol and the numpy array interface. This means that <code>numpy.asarray(this)</code> will yield a view of the original array, so mutations are visible on the original.</p>
        <p>Otherwise, the numpy <code>__array__</code> method is supported, and this returns an array of Python objects converted from the contents of the array. In this case, <code>numpy.asarray(this)</code> is a copy of the original array.</p>
        <h6>Members</h6>
        <ul>
            <li><code>ndim</code>: The number of dimensions.</li>
            <li><code>shape</code>: Tuple of lengths in each dimension.</li>
            <li><code>copy()</code>: A copy of the array.</li>
            <li><code>reshape(shape)</code>: A reshaped view of the array.</li>
        </ul>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="julia.DictValue" href="#julia.DictValue">julia.DictValue</a>
        —
        <span class="docstring-category">Python Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>AbstractDict</code> value. It is a subclass of <code>collections.abc.Mapping</code> and behaves similar to a Python <code>dict</code>.</p>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="julia.SetValue" href="#julia.SetValue">julia.SetValue</a>
        —
        <span class="docstring-category">Python Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>AbstractSet</code> value. It is a subclass of <code>collections.abc.Set</code> and behaves similar to a Python <code>set</code>.</p>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="julia.IOValue" href="#julia.IOValue">julia.IOValue</a>
        —
        <span class="docstring-category">Python Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>IO</code> value. It is a subclass of <code>io.IOBase</code> and behaves like Python files.</p>
        <p>There are also subtypes <code>RawIOValue</code>, <code>BufferedIOValue</code> and <code>TextIOValue</code>, which are subclasses of <code>io.RawIOBase</code> (unbuffered bytes), <code>io.BufferedIOBase</code> (buffered bytes) and <code>io.TextIOBase</code> (text).</p>
        <h6>Members</h6>
        <ul>
            <li><code>torawio()</code>: Convert to a <code>RawIOValue</code>, an un-buffered bytes file-like object. (See also <a href="../pythonjl/#Python.pyrawio"><code>pyrawio</code></a>.)
            <li><code>tobufferedio()</code>: Convert to a <code>BufferedIOValue</code>, an buffered bytes file-like object. Julia <code>IO</code> objects are converted to this by default. (See also <a href="../pythonjl/#Python.pybufferedio"><code>pybufferedio</code></a>.)
            <li><code>totextio()</code>: Convert to a <code>TextIOValue</code>, a text file-like object. (See also <a href="../pythonjl/#Python.pytextio"><code>pytextio</code></a>.)
        </ul>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="julia.ModuleValue" href="#julia.ModuleValue">julia.ModuleValue</a>
        —
        <span class="docstring-category">Python Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>Module</code> value.</p>
        <p>It is the same as <a href="#julia.AnyValue"><code>AnyValue</code></a> except for one additional convenience method:</p>
        <ul>
            <li><code>seval([module=self], code)</code>: Evaluates the given code (a string) in the given module.</li>
        </ul>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="julia.TypeValue" href="#julia.TypeValue">julia.TypeValue</a>
        —
        <span class="docstring-category">Python Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>Type</code> value.</p>
        <p>It is the same as <a href="#julia.AnyValue"><code>AnyValue</code></a> except that indexing is used to access Julia's "curly" syntax for specifying parametric types:</p>
        <pre><code class="language-python hljs"><span class="hljs-keyword">from</span> julia <span class="hljs-keyword">import</span> Main <span class="hljs-keyword">as</span> jl
<span class="hljs-comment"># equivalent to Vector{Int}() in Julia</span>
jl.Vector[jl.Int]()</code></pre>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="julia.RawValue" href="#julia.RawValue">julia.RawValue</a>
        —
        <span class="docstring-category">Python Class</span>
    </header>
    <section>
        <p>Wraps any Julia value with a rigid interface suitable for generic programming.</p>
        <p>This is very similar to <a href="#julia.AnyValue"><code>AnyValue</code></a> except that indexing, calling, etc. will always return a `RawValue`.</p>
        <p>Indexing with a tuple corresponds to indexing in Julia with multiple values. To index with a single tuple, it will need to be wrapped in another tuple.</p>
        <h6>Members</h6>
        <ul>
            <li><code>_jl_any()</code>: Convert to a <a href="#julia.AnyValue"><code>AnyValue</code></a> (or subclass). (See also <a href="../pythonjl/#Python.pyjl">pyjl</a>.)</li>
        </ul>
    </section>
</article>
```
