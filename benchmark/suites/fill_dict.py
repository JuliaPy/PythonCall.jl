from random import random

from .suite import SUITE

def fill_dict(n: int):
    x = {}
    
    for i in range(n):
        x[str(i)] = i + random()

    return x

SUITE.append(("fill_dict", fill_dict, (1_000,)))

def fill_dict_pydel(n: int):
    x = {}
    
    for i in range(n):
        x[str(i)] = i + random()

        del i

    return x

SUITE.append(("fill_dict_pydel", fill_dict_pydel, (1_000,)))
