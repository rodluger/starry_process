"""
Runs a single test. Called from the Makefile.

"""

import pytest
import sys

assert len(sys.argv) == 2
test_file = sys.argv[1]
log_file = test_file.replace(".py", ".tex")
with open(log_file, "w") as f:
    result = pytest.main([test_file]).value
    if result == 0:
        print(r"\testpassicon", file=f)
    else:
        print(r"\testfailicon", file=f)
