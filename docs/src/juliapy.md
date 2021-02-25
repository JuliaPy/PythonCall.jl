# The Python module `juliaaa`

For interactive or scripting use, the simplest way to get started is:

```python
from juliaaa import Main as jl
```

This loads a single variable `jl` (a [`ModuleValue`](#juliaaa.ModuleValue)) which represents the `Main` module in Julia, from which all of Julia's functionality is available.

If you are writing a package which uses Julia, then to avoid polluting the global `Main` namespace you should do:

```python
import juliaaa; jl = juliaaa.newmodule("SomeName");
```

Now you can do `jl.rand(jl.Bool, 5, 5)`, which is equivalent to `rand(Bool, 5, 5)` in Julia.

When a Python value is passed to Julia, then typically it will be converted according to [this table](../conversion/#Python-to-Julia) with `T=Any`. Sometimes a more specific type will be used, such as when assigning to an array whose element type is known.

When a Julia value is returned to Python, it will normally be converted according to [this table](../conversion/#Julia-to-Python).

## Wrapper types

Apart from a few fundamental immutable types (see [here](../conversion/#Julia-to-Python)), all Julia values are by default converted into Python to some [`AnyValue`](#juliaaa.AnyValue) object, which wraps the original value. Some types are converted to a subclass of [`AnyValue`](#juliaaa.AnyValue) which provides additional Python semantics --- e.g. Julia vectors are interpreted as Python sequences.

There is also a [`RawValue`](#juliaaa.RawValue) object, which gives a stricter "Julia-only" interface, documented below. These types all inherit from `ValueBase`:

- `ValueBase`
  - [`RawValue`](#juliaaa.RawValue)
  - [`AnyValue`](#juliaaa.AnyValue)
    - [`NumberValue`](#juliaaa.NumberValue)
      - `ComplexValue`
      - `RealValue`
        - `RationalValue`
        - `IntegerValue`
    - [`ArrayValue`](#juliaaa.ArrayValue)
      - `VectorValue`
    - [`DictValue`](#juliaaa.DictValue)
    - [`SetValue`](#juliaaa.SetValue)
    - [`IOValue`](#juliaaa.IOValue)
      - `RawIOValue`
      - `BufferedIOValue`
      - `TextIOValue`
    - [`ModuleValue`](#juliaaa.ModuleValue)
    - [`TypeValue`](#juliaaa.TypeValue)

```@raw html
<article class="docstring">
    <header>
        <a class="docstring-binding" id="juliaaa.AnyValue" href="#juliaaa.AnyValue">juliaaa.AnyValue</a>
        —
        <span class="docstring-category">Class</span>
    </header>
    <section>
        <p>Wraps any Julia object, giving it some basic Python semantics. Subtypes provide extra semantics.</p>
        <p>Supports <code>repr(x)</code>, <code>str(x)</code>, attributes (<code>x.attr</code>), calling (<code>x(a,b)</code>), iteration, comparisons, <code>len(x)</code>, <code>a in x</code>, <code>dir(x)</code>.</p>
        <p>Calling, indexing, attribute access, etc. will convert the result to a Python object according to <a href="../conversion/#Julia-to-Python">this table</a>. This is typically a builtin Python type (for immutables) or a subtype of <code>AnyValue</code>.</p>
        <p>Attribute access can be used to access Julia properties as well as normal class members. In the case of a name clash, the class member will take precedence. For convenience with Julia naming conventions, <code>_b</code> at the end of an attribute is replaced with <code>!</code> and <code>_bb</code> is replaced with <code>!!</code>.</p>
        <h6>Members</h6>
        <ul>
            <li><code>_jl_raw()</code>: Convert to a <a href="#juliaaa.RawValue"><code>RawValue</code></a></li>
            <li><code>_jl_display()</code>: Display the object using Julia's display mechanism.</li>
            <li><code>_jl_help()</code>: Display help for the object.</li>
        </ul>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="juliaaa.NumberValue" href="#juliaaa.NumberValue">juliaaa.NumberValue</a>
        —
        <span class="docstring-category">Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>Number</code> value. It is a subclass of <code>numbers.Number</code> and behaves similar to other Python numbers.</p>
        <p>There are also subtypes <code>ComplexValue</code>, <code>RealValue</code>, <code>RationalValue</code>, <code>IntegerValue</code> which wrap values of the corresponding Julia types, and are subclasses of the corresponding <code>numbers</code> ABC.</p>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="juliaaa.ArrayValue" href="#juliaaa.ArrayValue">juliaaa.ArrayValue</a>
        —
        <span class="docstring-category">Class</span>
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
        <a class="docstring-binding" id="juliaaa.DictValue" href="#juliaaa.DictValue">juliaaa.DictValue</a>
        —
        <span class="docstring-category">Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>AbstractDict</code> value. It is a subclass of <code>collections.abc.Mapping</code> and behaves similar to a Python <code>dict</code>.</p>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="juliaaa.SetValue" href="#juliaaa.SetValue">juliaaa.SetValue</a>
        —
        <span class="docstring-category">Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>AbstractSet</code> value. It is a subclass of <code>collections.abc.Set</code> and behaves similar to a Python <code>set</code>.</p>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="juliaaa.IOValue" href="#juliaaa.IOValue">juliaaa.IOValue</a>
        —
        <span class="docstring-category">Class</span>
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
        <a class="docstring-binding" id="juliaaa.ModuleValue" href="#juliaaa.ModuleValue">juliaaa.ModuleValue</a>
        —
        <span class="docstring-category">Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>Module</code> value.</p>
        <p>It is the same as <a href="#juliaaa.AnyValue"><code>AnyValue</code></a> except for one additional convenience method:</p>
        <ul>
            <li><code>seval([module=self], code)</code>: Evaluates the given code (a string) in the given module.</li>
        </ul>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="juliaaa.TypeValue" href="#juliaaa.TypeValue">juliaaa.TypeValue</a>
        —
        <span class="docstring-category">Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>Type</code> value.</p>
        <p>It is the same as <a href="#juliaaa.AnyValue"><code>AnyValue</code></a> except that indexing is used to access Julia's "curly" syntax for specifying parametric types:</p>
        <pre><code class="language-python hljs"><span class="hljs-keyword">from</span> juliaaa <span class="hljs-keyword">import</span> Main <span class="hljs-keyword">as</span> jl
<span class="hljs-comment"># equivalent to Vector{Int}() in Julia</span>
jl.Vector[jl.Int]()</code></pre>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="juliaaa.RawValue" href="#juliaaa.RawValue">juliaaa.RawValue</a>
        —
        <span class="docstring-category">Class</span>
    </header>
    <section>
        <p>Wraps any Julia value with a rigid interface suitable for generic programming.</p>
        <p>Supports <code>repr(x)</code>, <code>str(x)</code>, attributes (<code>x.attr</code>), calling (<code>x(a,b)</code>), <code>len(x)</code>, <code>dir(x)</code>.</p>
        <p>This is very similar to <a href="#juliaaa.AnyValue"><code>AnyValue</code></a> except that indexing, calling, etc. will always return a <code>RawValue</code>.</p>
        <p>Indexing with a tuple corresponds to indexing in Julia with multiple values. To index with a single tuple, it will need to be wrapped in another tuple.</p>
        <h6>Members</h6>
        <ul>
            <li><code>_jl_any()</code>: Convert to a <a href="#juliaaa.AnyValue"><code>AnyValue</code></a> (or subclass). (See also <a href="../pythonjl/#Python.pyjl">pyjl</a>.)</li>
        </ul>
    </section>
</article>
```

## Utilities

```@raw html
<article class="docstring">
    <header>
        <a class="docstring-binding" id="juliaaa.newmodule" href="#juliaaa.newmodule">juliaaa.newmodule</a>
        —
        <span class="docstring-category">Function</span>
    </header>
    <section>
        <pre><code class="language-python hljs">newmodule(name)</code></pre>
        <p>A new module with the given name.</p>
    </section>
</article>

<article class="docstring">
    <header>
        <a class="docstring-binding" id="juliaaa.As" href="#juliaaa.As">juliaaa.As</a>
        —
        <span class="docstring-category">Class</span>
    </header>
    <section>
        <pre><code class="language-python hljs">As(x, T)</code></pre>
        <p>When passed as an argument to a Julia function, is interpreted as <code>x</code> converted to Julia type <code>T</code>.</p>
    </section>
</article>
```
