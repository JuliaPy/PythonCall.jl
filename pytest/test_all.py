def test_import():
    import juliacall

def test_seval():
    from juliacall import Main

    Main.seval("""
    function f(x)
        return x^2
    end
    """)
