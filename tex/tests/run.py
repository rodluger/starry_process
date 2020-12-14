"""
Runs a single test. Called from the Makefile.

"""

import pytest
import sys

# File names
assert len(sys.argv) == 2
test_file = sys.argv[1]
log_file = test_file.replace(".py", ".tex")

# Run test & log result
result = pytest.main([test_file]).value
with open(log_file, "w") as f:
    if result == 0:
        print(r"\testpassicon", file=f)
    else:
        print(r"\testfailicon", file=f)
