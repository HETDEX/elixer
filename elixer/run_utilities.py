"""
usually have problems with linux command line expansion getting too long for list, find, rm, etc
so this is dumb little, directly editable script to use Python to do what you want
"""

import glob
import os
import sys
import time
import numpy as np

files = glob.glob(os.path.join(".","d??/dispatch_*/d*/*.pdf"))
bn = [os.path.basename(f).rstrip(".pdf") for f in files]
np.savetxt("all_pdf.list",bn,fmt="%s")

files = glob.glob(os.path.join(".","r???_???/*[0-9].png"))
bn = [os.path.basename(f).rstrip(".png") for f in files]
np.savetxt("all_png.list",bn,fmt="%s")