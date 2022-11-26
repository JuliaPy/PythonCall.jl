using SnoopPrecompile

@precompile_setup begin
    @precompile_all_calls begin
        withenv(
            # "JULIA_PYTHONCALL_LIBSTDCXX_VERSION_BOUND" => nothing,
            # "JULIA_PYTHONCALL_LIBPTR" => nothing,
            # "JULIA_PYTHONCALL_LIB" => nothing,
            # "JULIA_PYTHONCALL_EXE" => nothing,
        ) do
            C.init_context()
            C.with_gil() do
                C.init_jlwrap()
                init_consts()
                # init_pyconvert()  # FIXME: segfaults
                init_datetime()
                # juliacall/jlwrap
                init_juliacall()
                init_jlwrap_base()
                init_jlwrap_raw()
                init_jlwrap_callback()
                init_jlwrap_any()
                init_jlwrap_module()
                init_jlwrap_type()
                init_jlwrap_iter()
                init_jlwrap_array()
                init_jlwrap_vector()
                init_jlwrap_dict()
                init_jlwrap_set()
                init_jlwrap_number()
                init_jlwrap_io()
                init_juliacall_2()
                # compat
                # init_stdlib()  # FIXME: segfaults
                init_pyshow()
                # init_gui()  # FIXME: segfaults
                init_tables()
                init_ctypes()
                init_numpy()
                init_pandas()
            end
            C.CTX[] = C.Context()
        end
    end
end
