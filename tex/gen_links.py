from __future__ import print_function
import subprocess
import os

# Generate the github links
hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8")[
    :-1
]
slug = "rodluger/starry_gp"
with open("gitlinks.tex", "w") as f:
    print(
        r"\newcommand{\codelink}[1]{\href{https://github.com/%s/blob/%s/tex/figures/#1.py}{\codeicon}\,\,}"
        % (slug, hash),
        file=f,
    )
    print(
        r"\newcommand{\animlink}[1]{\href{https://github.com/%s/blob/%s/tex/figures/#1.gif}{\animicon}\,\,}"
        % (slug, hash),
        file=f,
    )
    print(
        r"\newcommand{\prooflink}[1]{\href{https://github.com/%s/blob/%s/tex/tests/#1.py}{\raisebox{-0.1em}{\input{tests/#1.tex}}}}"
        % (slug, hash),
        file=f,
    )
    print(
        r"\newcommand{\cilink}[1]{\href{https://dev.azure.com/%s/_build}{#1}}"
        % (slug),
        file=f,
    )
