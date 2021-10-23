# Compatibility Tools

Some packages require a little extra help to work nicely with PythonCall.

Some of these are "fixes" that are silently applied for you, and some are just extra functions to bridge a gap. We aim to keep these as minimal as possible.

## Stdlib

Whenever a Python exception is displayed by Julia, `sys.last_traceback` and friends are set. This allows the post-mortem debugger `pdb.pm()` to work. Disable by setting `PythonCall.CONFIG.auto_sys_last_traceback = false`.

## Tabular data & Pandas

A `pandas.DataFrame` can be wrapped in Julia as a [`PyPandasDataFrame`](@ref), providing a `Tables.jl`-compatible interface.

Furthermore, any Python object which can be converted to a `PyTable` (e.g. `pandas.DataFrame` can be converted to `PyPandasDataFrame`) satisfies the Tables.jl interface.

In the other direction, the following functions can be used to convert any `Tables.jl`-compatible table to a Python table.

```@docs
pytable
```

## MatPlotLib / PyPlot / Seaborn

MatPlotLib figures can be shown with Julia's display mechanism,
like `display(fig)` or `display(mime, fig)`.

This means that if you return a figure from a Jupyter or Pluto notebook cell,
it will be shown. You can call `display(plt.gcf())` to display the current figure.

We also provide a simple MatPlotLib backend: `mpl.use("module://juliacall.matplotlib")`.
Now you can call `plt.show()` to display the figure with Julia's display mechanism.
You can specify the format like `plt.show(format="png")`.

## GUIs (including MatPlotLib)

### Event loops

If for example you wish to use PyPlot in interactive mode (`matplotlib.pyplot.ion()`)
then activating the correct event loop will allow it to work.

```@docs
PythonCall.event_loop_on
PythonCall.event_loop_off
```

### Qt path fix

```@docs
PythonCall.fix_qt_plugin_path
```

## IPython

If Python is running an IPython kernel, then:
- Julia's `Base.stdout` is set to Python's `sys.stdout`.
- An `IPythonDisplay` is pushed onto Julia's display stack, so that `display(x)` goes to IPython if possible.

This is disabled by setting `PythonCall.CONFIG.auto_ipython_display = false`.
