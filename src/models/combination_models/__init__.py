# Import reload from importlib
from importlib import reload

# Import and reload some files
from . import utils
reload(utils)

# Import the define model function from the file utils.py
from .utils import define_model