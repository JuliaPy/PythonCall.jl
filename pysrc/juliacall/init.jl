try
    import Pkg
    Pkg.activate(ENV["JULIA_PYTHONCALL_PROJECT"], io=devnull)
    import PythonCall
catch err
    print(stderr, "ERROR: ")
    showerror(stderr, err, catch_backtrace())
    flush(stderr)
    rethrow()
end
