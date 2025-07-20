if Base.VERSION ≥ v"1.11"
    eval(Meta.parse("""
        public
            GC,
            GIL,
            VERSION,

            # C
            python_executable_path,
            python_library_handle,
            python_library_path,
            python_version,

            # Core
            CONFIG,
            getptr,
            pycopy!,
            pydel!,
            pyisnull,
            pynew,
            PyNULL,
            unsafe_pynext,
            
            # Compat
            event_loop_off,
            event_loop_on,
            fix_qt_plugin_path
    """))
end
