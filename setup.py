import site, sys, setuptools

if __name__ == '__main__':
    # https://github.com/pypa/pip/issues/7953#issuecomment-645133255
    site.ENABLE_USER_SITE = "--user" in sys.argv[1:]
    setuptools.setup()