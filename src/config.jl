Base.@kwdef mutable struct Config
    sysautolasttraceback :: Bool = true
end

const CONFIG = Config()
