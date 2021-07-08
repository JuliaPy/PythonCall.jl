"""
    fix_qt_plugin_path()

Try to set the `QT_PLUGIN_PATH` environment variable in Python, if not already set.

This fixes the problem that Qt does not know where to find its `qt.conf` file, because it
always looks relative to `sys.executable`, which can be the Julia executable not the Python
one when using this package.

If `CONFIG.qtfix` is true, then this is run automatically before `PyQt4`, `PyQt5`, `PySide` or `PySide2` are imported.
"""
function fix_qt_plugin_path()
    CONFIG.exepath === nothing && return false
    e = pyosmodule().environ
    "QT_PLUGIN_PATH" in e && return false
    qtconf = joinpath(dirname(CONFIG.exepath), "qt.conf")
    isfile(qtconf) || return false
    for line in eachline(qtconf)
        m = match(r"^\s*prefix\s*=(.*)$"i, line)
        if m !== nothing
            path = strip(m.captures[1])
            path[1] == path[end] == '"' && (path = path[2:end-1])
            path = joinpath(path, "plugins")
            if isdir(path)
                e["QT_PLUGIN_PATH"] = realpath(path)
                return true
            end
        end
    end
    return false
end

"""
    pyinteract(; force=false, sleep=0.1)

Some Python GUIs can work interactively, meaning the GUI is available but the interactive prompt is returned (e.g. after calling `matplotlib.pyplot.ion()`).
To use these from Julia, currently you must manually call `pyinteract()` each time you want to interact.

Internally, this is calling the `PyOS_InputHook` asynchronously. Only one copy is run at a time unless `force` is true.

The asynchronous task waits for `sleep` seconds before calling the hook function.
This gives time for the next prompt to be printed and waiting for input.
As a result, there will be a small delay before the GUI becomes interactive.
"""
pyinteract(; force::Bool = false, sleep::Real = 0.1) =
    if !CONFIG.inputhookrunning || force
        CONFIG.inputhookrunning = true
        @async begin
            sleep > 0 && Base.sleep(sleep)
            C.PyOS_RunInputHook()
            CONFIG.inputhookrunning = false
        end
        nothing
    end
export pyinteract

const EVENT_LOOPS = Dict{Symbol,Base.Timer}()

"""
    event_loop_off(g::Symbol)

Terminate the event loop `g` if it is running.
"""
function event_loop_off(g::Symbol)
    if haskey(EVENT_LOOPS, g)
        Base.close(pop!(EVENT_LOOPS, g))
    end
    return
end

"""
    event_loop_on(g::Symbol; interval=40e-3, fix=false)

Activate an event loop for the GUI framework `g`, so that the framework can run in the background of a Julia session.

The event loop runs every `interval` seconds. If `fix` is true and `g` is a Qt framework, then [`fix_qt_plugin_path`](@ref) is called.

Supported values of `g` (and the Python module they relate to) are: `:pyqt4` (PyQt4), `:pyqt5` (PyQt5), `:pyside` (PySide), `:pyside2` (PySide2), `:gtk` (gtk), `:gtk3` (gi), `:wx` (wx), `:tkinter` (tkinter).
"""
function event_loop_on(g::Symbol; interval::Real = 40e-3, fix::Bool = false)
    haskey(EVENT_LOOPS, g) && return EVENT_LOOPS[g]
    fix && g in (:pyqt4, :pyqt5, :pyside, :pyside2) && fix_qt_plugin_path()
    @py ```
    def make_event_loop(g, interval):
        if g in ("pyqt4","pyqt5","pyside","pyside2"):
            if g == "pyqt4":
                import PyQt4.QtCore as QtCore
            elif g == "pyqt5":
                import PyQt5.QtCore as QtCore
            elif g == "pyside":
                import PySide.QtCore as QtCore
            elif g == "pyside2":
                import PySide2.QtCore as QtCore
            instance = QtCore.QCoreApplication.instance
            AllEvents = QtCore.QEventLoop.AllEvents
            processEvents = QtCore.QCoreApplication.processEvents
            maxtime = interval * 1000
            def event_loop():
                app = instance()
                if app is not None:
                    app._in_event_loop = True
                    processEvents(AllEvents, maxtime)
        elif g in ("gtk","gtk3"):
            if g == "gtk3":
                import gi
                if gi.get_required_version("Gtk") is None:
                    gi.require_version("Gtk", "3.0")
                import gi.repository.Gtk as gtk
            elif g == "gtk":
                import gtk
            events_pending = gtk.events_pending
            main_iteration = gtk.main_iteration
            def event_loop():
                while events_pending():
                    main_iteration()
        elif g == "wx":
            import wx
            GetApp = wx.GetApp
            EventLoop = wx.EventLoop
            EventLoopActivator = wx.EventLoopActivator
            def event_loop():
                app = GetApp()
                if app is not None:
                    app._in_event_loop = True
                    evtloop = EventLoop()
                    ea = EventLoopActivator(evtloop)
                    Pending = evtloop.Pending
                    Dispatch = evtloop.Dispatch
                    while Pending():
                        Dispatch()
                    app.ProcessIdle()
        elif g == "tkinter":
            import tkinter, _tkinter
            flag = _tkinter.ALL_EVENTS | _tkinter.DONT_WAIT
            root = None
            def event_loop():
                global root
                new_root = tkinter._default_root
                if new_root is not None:
                    root = new_root
                if root is not None:
                    while root.dooneevent(flag):
                        pass
        else:
            raise ValueError("invalid event loop name: {}".format(g))
        return event_loop
    $event_loop = make_event_loop($(string(g)), $interval)
    ```
    EVENT_LOOPS[g] = Timer(t -> event_loop(), 0; interval = interval)
end
