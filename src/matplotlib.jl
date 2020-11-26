pymatplotlib = PyLazyObject(() -> pyimport("matplotlib"))
pyplot = PyLazyObject(() -> pyimport("matplotlib.pyplot"))

"""
    pyplotshow([fig]; close=true)

Show the matplotlib figure `fig` (or the current figure if not given).

If `close` is true, the figure is also closed.
"""
function pyplotshow(fig=pyplot.gcf(); close::Bool=true)
    fig = pyisinstance(fig, pyplot.Figure) ? PyObject(fig) : pyplot.figure(fig)
    if displayable(MIME("text/html"))
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
export pyplotshow
