if __name__ == '__main__':
    import juliacall as _  # calls init() which calls juliapkg.executable() which lazily downloads julia

    import sys
    if "--debug" in sys.argv:
        import juliapkg, json
        state = juliapkg.state.STATE
        state["version"] = str(state["version"])
        print(json.dumps(state, indent=2))
    else:
        print("Initialized successfully. Pass --debug to see the full JuliaPkg state.")