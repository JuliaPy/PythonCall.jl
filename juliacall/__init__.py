__version__ = '0.5.1'
CONFIG = dict(embedded=False)
from .all import *

# In application code, use `juliacall.default_init()`
#from . import init
from .init import default_init
