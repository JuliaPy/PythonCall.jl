"""
    fix_qt_plugin_path()

Try to set the `QT_PLUGIN_PATH` environment variable in Python, if not already set.

This fixes the problem that Qt does not know where to find its `qt.conf` file, because it
always looks relative to `sys.executable`, which can be the Julia executable not the Python
one when using this package.
"""
function fix_qt_plugin_path()
    e = pyosmodule.environ
    "QT_PLUGIN_PATH" in e && return false
    CONFIG.exepath === nothing && return false
    qtconf = joinpath(dirname(CONFIG.exepath), "qt.conf")
    isfile(qtconf) || return false
    for line in eachline(qtconf)
        m = match(r"^\s*prefix\s*=(.*)$"i, line)
        if m !== nothing
            path = strip(m.captures[1])
            path[1]==path[end]=='"' && (path = path[2:end-1])
            path = joinpath(path, "plugins")
            if isdir(path)
                e["QT_PLUGIN_PATH"] = path
                return true
            end
        end
    end
    return false
end

pyisimportable(modname) = try; pyimport(modname); true; catch; false; end

const ALL_GUI_MODULES = ["wx", "gtk", "gi", "tkinter", "Tkinter", "PyQt4", "PyQt5", "PySide", "PySide2"]

find_gui_modules(q::String="any") =
    if q in ALL_GUI_MODULES
        filter(pyisimportable, [String(q)])
    elseif q == "GTK"
        filter(pyisimportable, ["gtk", "gi"])
    elseif q == "Tk"
        filter(pyisimportable, ["tkinter", "Tkinter"])
    elseif q == "Qt4"
        filter(pyisimportable, ["PyQt4", "PySide"])
    elseif q == "Qt5"
        filter(pyisimportable, ["PyQt5", "PySide2"])
    elseif q == "Qt"
        filter(pyisimportable, ["PyQt4", "PySide", "PyQt5", "PySide2"])
    elseif q == "any"
        filter(pyisimportable, ALL_GUI_MODULES)
    else
        String[]
    end

const EVENT_LOOPS = Dict{String,Base.Timer}()

function stop_event_loop(modname::String; force::Bool=false)
    if haskey(EVENT_LOOPS, modname)
        Base.close(pop!(EVENT_LOOPS, modname))
        return
    elseif force
        return
    else
        error("No such event loop $(repr(modname)).")
    end
end

function ensure_no_event_loop(modname::String, force::Bool=false)
    if haskey(EVENT_LOOPS, modname)
        if force
            stop_event_loop(modname)
        else
            error("Event loop $(repr(modname)) already running.")
        end
    end
    return
end

function start_qt_event_loop(modname::String; freq::Real=50e-3, force::Bool=false, fix::Bool=false)
    ensure_no_event_loop(modname, force)
    fix_qt_plugin_path()
    mod = pyimport("$modname.QtCore")
    instance = mod.QCoreApplication.instance
    AllEvents = mod.QEventLoop.AllEvents
    processEvents = mod.QCoreApplication.processEvents
    maxtime = pyobject(1000*freq)
    EVENT_LOOPS[modname] = Timer(0; interval=freq) do t
        app = instance()
        if !pyisnone(app)
            app._in_event_loop = true
            processEvents(AllEvents, maxtime)
        end
    end
end

function start_gtk_event_loop(modname::String; freq::Real=50e-3, force::Bool=false, fix::Bool=false)
    ensure_no_event_loop(modname, force)
    if modname == "gi"
        mod = pyimport("gi.repository.Gtk")
        if fix
            gi = pyimport("gi")
            if pyisnone(gi.get_required_version("Gtk"))
                gi.require_version("Gtk", "3.0")
            end
        end
    else
        mod = pyimport(modname)
    end
    events_pending = mod.events_pending
    main_iteration = mod.main_iteration
    EVENT_LOOPS[modname] = Timer(0; interval=freq) do t
        while pytruth(events_pending())
            main_iteration()
        end
    end
end

function start_wx_event_loop(modname::String; freq::Real=50e-3, force::Bool=false)
    ensure_no_event_loop(modname, force)
    mod = pyimport(modname)
    GetApp = mod.GetApp
    EventLoop = mod.EventLoop
    EventLoopActivator = mod.EventLoopActivator
    EVENT_LOOPS[modname] = Timer(0; interval=freq) do t
        app = GetApp()
        if !pyisnone(app)
            app._in_event_loop = true
            evtloop = EventLoop()
            ea = EventLoopActivator(evtloop)
            Pending = evtloop.Pending
            Dispatch = evtloop.Dispatch
            while pytruth(Pending())
                Dispatch()
            end
            finalize(ea) # deactivate event loop
            app.ProcessIdle()
        end
    end
end

function start_tk_event_loop(modname::String; freq::Real=50e-3, force::Bool=false)
    ensure_no_event_loop(modname, force)
    mod = pyimport(modname)
    _tkinter = pyimport("_tkinter")
    flag = _tkinter.ALL_EVENTS | _tkinter.DONT_WAIT
    root = PyObject(pynone)
    EVENT_LOOPS[modname] = Timer(0; interval=freq) do t
        new_root = mod._default_root
        if !pyisnone(new_root)
            root = new_root
        end
        if !pyisnone(root)
            while pytruth(root.dooneevent(flag))
            end
        end
    end
end

function start_event_loop(q::String="any"; opts...)
    ms = find_gui_modules(q)
    if isempty(ms)
        error("No such GUI library: $(repr(q))")
    elseif length(ms) == 1
        m = ms[1]
        if m in ("PyQt4", "PyQt5", "PySide", "PySide2")
            start_qt_event_loop(m; opts...)
        elseif m in ("gtk", "gi")
            start_gtk_event_loop(m; opts...)
        elseif m in ("wx",)
            start_wx_event_loop(m; opts...)
        elseif m in ("tkinter", "Tkinter")
            start_tk_event_loop(m; opts...)
        else
            error("not implemented: `start_event_loop($(repr(m)))`")
        end
    else
        error("Multiple GUI libraries matching this query: $(join(ms, ", "))")
    end
end
