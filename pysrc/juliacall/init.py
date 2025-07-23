import json
import juliapkg
import sys

if __name__ == '__main__':
    # invoking python -m juliacall.init automatically imports juliacall which
    # calls init() which calls juliapkg.executable() which lazily downloads julia

    if "--debug" in sys.argv:
        state = juliapkg.state.STATE
        state["version"] = str(state["version"])
        print(json.dumps(state, indent=2))
    else:
        print("Initialized successfully. Pass --debug to see the full JuliaPkg state.")