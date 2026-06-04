@kwdef mutable struct Config
    meta::String = ""
    auto_fix_qt_plugin_path::Bool = true
end

const CONFIG = Config()
