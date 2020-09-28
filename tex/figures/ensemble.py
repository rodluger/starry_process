from static.sample import sample
import os


# DEBUG
if not int(os.getenv("ON_AZURE", "0")):
    sample(1)
