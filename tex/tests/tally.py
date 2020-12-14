import glob

test_files = glob.glob("test_*.tex")

passed = 0
failed = 0
for file in test_files:
    with open(file, "r") as f:
        l = f.readline()
    if r"\testpassicon" in l:
        passed += 1
    elif r"\testfailicon" in l:
        failed += 1

with open("tally.tex", "w") as f:
    print(
        r"{} passed \testpassicon, {} failed \testfailicon".format(
            passed, failed
        ),
        file=f,
    )
