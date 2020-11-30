pymatplotlib = PyLazyObject(() -> pyimport("matplotlib"))
pyplot = PyLazyObject(() -> pyimport("matplotlib.pyplot"))

"""
    pyplotshow([fig]; close=true)

Show the matplotlib/pyplot/seaborn/etc figure `fig`, or all open figures if not given.

If `close` is true, the figure is also closed.

If `CONFIG.pyplotautoshow` is true, then this is automatically called each time a notebook cell is evaluated.
"""
function pyplotshow(fig; close::Bool=true)
    fig = pyisinstance(fig, pyplot.Figure) ? PyObject(fig) : pyplot.figure(fig)
    if displayable(MIME("image/png"))
        buf = IOBuffer()
        fig.savefig(buf, format="png")
        data = take!(buf)
        display(MIME("image/png"), data)
    elseif displayable(MIME("text/html"))
        buf = IOBuffer()
        fig.savefig(buf, format="png")
        data = base64encode(take!(buf))
        display(MIME("text/html"), HTML("""<img src="data:image/png;base64,$(data)" />"""))
    else
        display(fig)
    end
    close && pyplot.close(fig)
    nothing
end
function pyplotshow(; opts...)
    for fig in pyplot.get_fignums()
        pyplotshow(fig; opts...)
    end
end
export pyplotshow
