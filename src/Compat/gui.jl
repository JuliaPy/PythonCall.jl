"""
    fix_qt_plugin_path()

Try to set the `QT_PLUGIN_PATH` environment variable in Python, if not already set.

This fixes the problem that Qt does not know where to find its `qt.conf` file, because it
always looks relative to `sys.executable`, which can be the Julia executable not the Python
one when using this package.

If `CONFIG.auto_fix_qt_plugin_path` is true, then this is run automatically before `PyQt4`, `PyQt5`, `PySide` or `PySide2` are imported.
"""
function fix_qt_plugin_path()
    C.CTX.exe_path === nothing && return false
    e = pyosmodule.environ
    "QT_PLUGIN_PATH" in e && return false
    qtconf = joinpath(dirname(C.CTX.exe_path::AbstractString), "qt.conf")
    isfile(qtconf) || return false
    for line in eachline(qtconf)
        m = match(r"^\s*prefix\s*=(.*)$"i, line)
        if m !== nothing
            path = strip(m.captures[1]::AbstractString)
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

# """
#     pyinteract(; force=false, sleep=0.1)

# Some Python GUIs can work interactively, meaning the GUI is available but the interactive prompt is returned (e.g. after calling `matplotlib.pyplot.ion()`).
# To use these from Julia, currently you must manually call `pyinteract()` each time you want to interact.

# Internally, this is calling the `PyOS_InputHook` asynchronously. Only one copy is run at a time unless `force` is true.

# The asynchronous task waits for `sleep` seconds before calling the hook function.
# This gives time for the next prompt to be printed and waiting for input.
# As a result, there will be a small delay before the GUI becomes interactive.
# """
# pyinteract(; force::Bool = false, sleep::Real = 0.1) =
#     if !CONFIG.inputhookrunning || force
#         CONFIG.inputhookrunning = true
#         @async begin
#             sleep > 0 && Base.sleep(sleep)
#             C.PyOS_RunInputHook()
#             CONFIG.inputhookrunning = false
#         end
#         nothing
#     end
# export pyinteract

const EVENT_LOOPS = Dict{Symbol,Base.Timer}()

const new_event_loop_callback = pynew()

function init_gui()
    if !C.CTX.is_embedded
        # define callbacks
        g = pydict()
        pyexec(
            """
     def new_event_loop_callback(g, interval=0.04):
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
             maxtime = int(interval * 1000)
             def callback():
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
             def callback():
                 while events_pending():
                     main_iteration()
         elif g == "wx":
             import wx
             GetApp = wx.GetApp
             EventLoop = wx.EventLoop
             EventLoopActivator = wx.EventLoopActivator
             def callback():
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
             def callback():
                 global root
                 new_root = tkinter._default_root
                 if new_root is not None:
                     root = new_root
                 if root is not None:
                     while root.dooneevent(flag):
                         pass
         else:
             raise ValueError("invalid event loop name: {}".format(g))
         return callback
     """,
            g,
        )
        pycopy!(new_event_loop_callback, g["new_event_loop_callback"])

        # add a hook to automatically call fix_qt_plugin_path()
        fixqthook = Py(
            () -> (
                PythonCall.CONFIG.auto_fix_qt_plugin_path && fix_qt_plugin_path(); nothing
            ),
        )
        pymodulehooks.add_hook("PyQt4", fixqthook)
        pymodulehooks.add_hook("PyQt5", fixqthook)
        pymodulehooks.add_hook("PySide", fixqthook)
        pymodulehooks.add_hook("PySide2", fixqthook)
    end
end

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
    event_loop_on(g::Symbol; interval=0.04, fix=false)

Activate an event loop for the GUI framework `g`, so that the framework can run in the background of a Julia session.

The event loop runs every `interval` seconds. If `fix` is true and `g` is a Qt framework, then [`fix_qt_plugin_path`](@ref PythonCall.fix_qt_plugin_path) is called.

Supported values of `g` (and the Python module they relate to) are: `:pyqt4` (PyQt4), `:pyqt5` (PyQt5), `:pyside` (PySide), `:pyside2` (PySide2), `:gtk` (gtk), `:gtk3` (gi), `:wx` (wx), `:tkinter` (tkinter).
"""
function event_loop_on(g::Symbol; interval::Real = 0.04, fix::Bool = false)
    haskey(EVENT_LOOPS, g) && return EVENT_LOOPS[g]
    fix && g in (:pyqt4, :pyqt5, :pyside, :pyside2) && fix_qt_plugin_path()
    callback = new_event_loop_callback(string(g), Float64(interval))
    EVENT_LOOPS[g] = Timer(t -> callback(), 0; interval = interval)
end

function _python_input_hook()
    try
        @static if Sys.iswindows()
            # on windows, we can call yield in a loop because _kbhit() lets us know
            # when to stop
            while true
                yield()
                if ccall(:_kbhit, Cint, ()) != 0
                    break
                end
                sleep(0.01)
            end
        else
            # on other platforms, if readline is enabled, the input hook is called
            # repeatedly so the loop is not required
            yield()
        end
    catch
        return Cint(1)
    end
    return Cint(0)
end

function _set_python_input_hook()
    C.PyOS_SetInputHook(@cfunction(_python_input_hook, Cint, ()))
    return
end

function _unset_python_input_hook()
    C.PyOS_SetInputHook(C_NULL)
    return
end
