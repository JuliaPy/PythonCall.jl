# https://github.com/JuliaLang/julia/blob/fae0d0ad3e5d9804533435fe81f4eaac819895af/stdlib/REPL/src/REPL.jl#L1727C1-L1795C4

function __PythonCall_banner(banner_opt::Symbol = :yes)
    io = stdout

    if banner_opt == :no
        return
    end

    short = banner_opt == :short

    if get(io, :color, false)::Bool
        c = Base.text_colors
        tx = c[:normal] # text
        jl = c[:normal] # julia
        jc = c[:blue] # juliacall text
        jb = c[:bold] * jc  # bold blue dot
        d1 = c[:bold] * c[:blue]    # first dot
        d2 = c[:bold] * c[:red]     # second dot
        d3 = c[:bold] * c[:green]   # third dot
        d4 = c[:bold] * c[:magenta] # fourth dot
        d5 = c[:bold] * c[:yellow]  # bold yellow dot

        if short
            print(io,"""
              $(jb)o$(tx)  | Julia $(VERSION)
             $(jb)o$(tx) $(d5)o$(tx) | PythonCall $(PythonCall.VERSION)
            """)
        else
            print(io,"""               $(d3)_$(tx)
               $(d1)_$(tx)       $(jl)_$(tx) $(d2)_$(d3)(_)$(d4)_$(tx)$(jc)               _  _ $(tx)  |  Documentation: https://juliapy.github.io/PythonCall.jl/
              $(d1)(_)$(jl)     | $(d2)(_)$(tx) $(d4)(_)$(tx)$(jc)             | || |$(tx)  |
               $(jl)_ _   _| |_  __ _$(jc)  ___  __ _ | || |$(tx)  |  Julia: $(VERSION)
              $(jl)| | | | | | |/ _` |$(jc)/ __|/ _` || || |$(tx)  |  PythonCall: $(PythonCall.VERSION)
              $(jl)| | |_| | | | (_| |$(jc) |__  (_| || || |$(tx)  |
             $(jl)_/ |\\__'_|_|_|\\__'_|$(jc)\\___|\\__'_||_||_|$(tx)  |  The JuliaCall REPL is experimental.
            $(jl)|__/$(tx)                                    |

            """)
        end
    else
        if short
            print(io,"""
              o  |  Julia $(VERSION)
             o o |  PythonCall $(PythonCall.VERSION)
            """)
        else
            print(io,"""
                           _
               _       _ _(_)_               _  _   |  Documentation: https://juliapy.github.io/PythonCall.jl/
              (_)     | (_) (_)             | || |  |
               _ _   _| |_  __ _  ___  __ _ | || |  |  Julia: $(VERSION)
              | | | | | | |/ _` |/ __|/ _` || || |  |  PythonCall: $(PythonCall.VERSION)
              | | |_| | | | (_| | |__  (_| || || |  |
             _/ |\\__'_|_|_|\\__'_|\\___|\\__'_||_||_|  |  The JuliaCall REPL is experimental.
            |__/                                    |
            """)
        end
    end
end
