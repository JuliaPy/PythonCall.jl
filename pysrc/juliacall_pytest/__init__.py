import os
import sys

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config):
    handle_signals = sys._xoptions.get("juliacall-handle-signals")
    if handle_signals is None:
        handle_signals = os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS")
    if sys.platform != "linux" or handle_signals != "yes":
        return
    if not early_config.pluginmanager.hasplugin("faulthandler"):
        return
    try:
        exit_on_timeout = early_config.getini("faulthandler_exit_on_timeout")
    except ValueError as error:
        if not isinstance(error.__cause__, KeyError) or error.__cause__.args != ("faulthandler_exit_on_timeout",):
            raise
        # Older pytest versions have no exit-on-timeout option.
        exit_on_timeout = False
    if exit_on_timeout and float(early_config.getini("faulthandler_timeout")) > 0:
        raise pytest.UsageError(
            "Julia signal handling conflicts with pytest's faulthandler_exit_on_timeout. "
            "Use an external process timeout instead."
        )
    early_config.pluginmanager.set_blocked("faulthandler")
    early_config.issue_config_time_warning(
        pytest.PytestConfigWarning(
            "Julia signal handling is enabled: disabling pytest's faulthandler plugin "
            "and its timeout diagnostics to preserve Julia's signal handlers. "
            "Julia fatal diagnostics remain enabled. Use -p no:juliacall to opt out."
        ),
        stacklevel=2,
    )
