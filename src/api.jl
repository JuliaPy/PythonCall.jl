"The version of PythonCall."
const VERSION = v"0.9.23"

# public functions
for name in
    [:python_executable_path, :python_library_handle, :python_library_path, :python_version]
    @eval function $name end
    if Base.VERSION ≥ v"1.11"
        eval(Expr(:public, name))
    end
end

# other public bindings
for name in [:VERSION]
    if Base.VERSION ≥ v"1.11"
        eval(Expr(:public, name))
    end
end
