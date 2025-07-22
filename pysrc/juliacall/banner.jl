# https://github.com/JuliaLang/julia/blob/fae0d0ad3e5d9804533435fe81f4eaac819895af/stdlib/REPL/src/REPL.jl#L1727C1-L1795C4

function banner(io::IO = stdout; short = false)
    if Base.GIT_VERSION_INFO.tagged_commit
        commit_string = Base.TAGGED_RELEASE_BANNER
    elseif isempty(Base.GIT_VERSION_INFO.commit)
        commit_string = ""
    else
        days = Int(floor((ccall(:jl_clock_now, Float64, ()) - Base.GIT_VERSION_INFO.fork_master_timestamp) / (60 * 60 * 24)))
        days = max(0, days)
        unit = days == 1 ? "day" : "days"
        distance = Base.GIT_VERSION_INFO.fork_master_distance
        commit = Base.GIT_VERSION_INFO.commit_short

        if distance == 0
            commit_string = "Commit $(commit) ($(days) $(unit) old master)"
        else
            branch = Base.GIT_VERSION_INFO.branch
            commit_string = "$(branch)/$(commit) (fork: $(distance) commits, $(days) $(unit))"
        end
    end

    commit_date = isempty(Base.GIT_VERSION_INFO.date_string) ? "" : " ($(split(Base.GIT_VERSION_INFO.date_string)[1]))"

    if get(io, :color, false)::Bool
        c = Base.text_colors
        tx = c[:normal] # text
        jl = c[:normal] # julia
        jc = c[:blue] # juliacall
        d1 = c[:bold] * c[:blue]    # first dot
        d2 = c[:bold] * c[:red]     # second dot
        d3 = c[:bold] * c[:green]   # third dot
        d4 = c[:bold] * c[:magenta] # fourth dot

        if short
            print(io,"""
              $(d3)o$(tx)  | Version $(VERSION)PythonCall: $(PythonCall.VERSION)
             $(d2)o$(tx) $(d4)o$(tx) | $(commit_string)
            """)
        else
            print(io,"""               $(d3)_$(tx)
               $(d1)_$(tx)       $(jl)_$(tx) $(d2)_$(d3)(_)$(d4)_$(tx)$(jc)               _  _ $(tx)  |  Documentation: https://juliapy.github.io/PythonCall.jl/
              $(d1)(_)$(jl)     | $(d2)(_)$(tx) $(d4)(_)$(tx)$(jc)             | || |$(tx)  |
               $(jl)_ _   _| |_  __ _$(jc)  ___  __ _ | || |$(tx)  |  Julia: $(VERSION)
              $(jl)| | | | | | |/ _` |$(jc)/ __|/ _` || || |$(tx)  |  PythonCall: $(PythonCall.VERSION)
              $(jl)| | |_| | | | (_| |$(jc) |__  (_| || || |$(tx)  |
             $(jl)_/ |\\__'_|_|_|\\__'_|$(jc)\\___|\\__'_||_||_|$(tx)  |  This REPL is running via Python using JuliaCall.
            $(jl)|__/$(tx)                                    |    Only basic features are supported.

            """)
        end
    else
        if short
            print(io,"""
              o  |  Version $(VERSION)PythonCall: $(PythonCall.VERSION)
             o o |  $(commit_string)
            """)
        else
            print(io,"""
                           _
               _       _ _(_)_               _  _   |  Documentation: https://juliapy.github.io/PythonCall.jl/
              (_)     | (_) (_)             | || |  |
               _ _   _| |_  __ _  ___  __ _ | || |  |  Julia: $(VERSION)
              | | | | | | |/ _` |/ __|/ _` || || |  |  PythonCall: $(PythonCall.VERSION)
              | | |_| | | | (_| | |__  (_| || || |  |
             _/ |\\__'_|_|_|\\__'_|\\___|\\__'_||_||_|  |  This REPL is running via Python using JuliaCall.
            |__/                                    |    Only basic features are supported.

            """)
        end
    end
end