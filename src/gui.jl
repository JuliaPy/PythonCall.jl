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
            path[1]==path[end]=='"' && (path = path[2:end-1])
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
pyinteract(; force::Bool=false, sleep::Real=0.1) =
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

const EVENT_LOOPS = Dict{Symbol, Base.Timer}()

function event_loop_off(g::Symbol)
    if haskey(EVENT_LOOPS, g)
        Base.close(pop!(EVENT_LOOPS, g))
    end
    return
end

# function event_loop_on(g::Symbol; interval::Real=40e-3, fix::Bool=false)
#     # check if already running
#     if haskey(EVENT_LOOPS, g)
#         return g => EVENT_LOOPS[g]
#     end
#     # start a new event loop
#     if g in (:pyqt4, :pyqt5, :pyside, :pyside2)
#         fix && fix_qt_plugin_path()
#         modname =
#             g == :pyqt4 ? "PyQt4" :
#             g == :pyqt5 ? "PyQt5" :
#             g == :pyside ? "PySide" :
#             g == :pyside2 ? "PySide2" : error()
#         mod = pyimport("$modname.QtCore")
#         instance = mod.QCoreApplication.instance
#         AllEvents = mod.QEventLoop.AllEvents
#         processEvents = mod.QCoreApplication.processEvents
#         maxtime = pyobject(1000*interval)
#         EVENT_LOOPS[g] = Timer(0; interval=interval) do t
#             app = instance()
#             if !pyisnone(app)
#                 app._in_event_loop = true
#                 processEvents(AllEvents, maxtime)
#             end
#         end
#     elseif g in (:gtk, :gtk3)
#         if g == :gtk3
#             gi = pyimport("gi")
#             if pyisnone(gi.get_required_version("Gtk"))
#                 gi.require_version("Gtk", "3.0")
#             end
#         end
#         mod = pyimport(g==:gtk ? "gtk" : g==:gtk3 ? "gi.repository.Gtk" : error())
#         events_pending = mod.events_pending
#         main_iteration = mod.main_iteration
#         EVENT_LOOPS[g] = Timer(0; interval=interval) do t
#             while pytruth(events_pending())
#                 main_iteration()
#             end
#         end
#     elseif g in (:wx,)
#         mod = pyimport("wx")
#         GetApp = mod.GetApp
#         EventLoop = mod.EventLoop
#         EventLoopActivator = mod.EventLoopActivator
#         EVENT_LOOPS[g] = Timer(0; interval=interval) do t
#             app = GetApp()
#             if !pyisnone(app)
#                 app._in_event_loop = true
#                 evtloop = EventLoop()
#                 ea = EventLoopActivator(evtloop)
#                 Pending = evtloop.Pending
#                 Dispatch = evtloop.Dispatch
#                 while pytruth(Pending())
#                     Dispatch()
#                 end
#                 finalize(ea) # deactivate event loop
#                 app.ProcessIdle()
#             end
#         end
#     elseif g in (:tkinter,)
#         mod = pyimport("tkinter")
#         _tkinter = pyimport("_tkinter")
#         flag = _tkinter.ALL_EVENTS | _tkinter.DONT_WAIT
#         root = PyObject(pynone)
#         EVENT_LOOPS[g] = Timer(0; interval=interval) do t
#             new_root = mod._default_root
#             if !pyisnone(new_root)
#                 root = new_root
#             end
#             if !pyisnone(root)
#                 while pytruth(root.dooneevent(flag))
#                 end
#             end
#         end
#     else
#         error("invalid gui: $(repr(g))")
#     end
#     g => EVENT_LOOPS[g]
# end
