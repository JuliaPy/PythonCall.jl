from juliacall import Main

RED = "\033[1;31m"
GREEN = "\033[1;32m"
RESET = "\033[0m"

import os
path_to_banner = os.path.join(os.path.dirname(__file__), "banner.jl")
Main.seval(f"include(\"{path_to_banner}\"); banner()")

while True:
    try:
        line = input(f"{GREEN}juliacall> {RESET}")
    except KeyboardInterrupt:
        print("\n")
        continue
    except EOFError:
        break
    if not line.strip():
        continue
    try:
        result = Main.seval(line)
        if result is not None:
            Main.display(result)
            print()
    except Exception as e:
        print(f"{RED}ERROR:{RESET} {e}")

