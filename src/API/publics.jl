if Base.VERSION ≥ v"1.11"
    eval(Meta.parse("""
        public VERSION,
               GIL,
               GC,
               # C
               python_executable_path,
               python_library_path,
               python_library_handle,
               python_version,
               # Core
               pynew,
               pyisnull,
               pycopy!,
               getptr,
               pydel!,
               unsafe_pynext,
               PyNULL,
               CONFIG,
               # Compat
               event_loop_on,
               event_loop_off,
               fix_qt_plugin_path
    """))
end
