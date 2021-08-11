Base.@kwdef mutable struct Config
    meta :: String = ""
    auto_sys_last_traceback :: Bool = true
    auto_fix_qt_plugin_path :: Bool = true
    auto_pyplot_show :: Bool = true
    auto_ipython_display :: Bool = true
end

const CONFIG = Config()

# TODO: load_config(), save_config()
