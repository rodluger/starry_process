.PHONY : clean, default, publish, pdflatex, tectonic

TEX_FILES   = ms
TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out ent
LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX      = bibtex
SUFF        = pdf
RM_TMP      = $(foreach d, ${TEX_FILES}, rm -rf $(foreach suff, ${TMP_SUFFS}, ${d}.${suff})) msNotes.bib
CHECK_RERUN = grep Rerun ms.log
FIGURES     = $(patsubst %.py,%.pdf,$(sort $(wildcard figures/*.py)))
TESTS       = $(patsubst %.py,%.tex,$(sort $(wildcard tests/test_*.py)))

default: tectonic

tectonic: ms.tex bib.bib $(FIGURES) $(TESTS)

	# Generate links to current git commit
	python gen_links.py

	# Compile the paper
	tectonic ms.tex --print --keep-logs


pdflatex: ms.tex bib.bib $(FIGURES) $(TESTS)

	# Generate links to current git commit
	python gen_links.py

	# Compile the paper
	${LATEX} ms.tex
	${BIBTEX} ms.aux
	${LATEX} ms.tex
	( ${CHECK_RERUN} && ${LATEX} ms.tex ) || echo "Done."

# Remove temporaries and output
clean:
	$(RM_TMP)
	rm -rf $(FIGURES)
	rm -rf $(TESTS)

# Generate all figures
figures/%.pdf: figures/%.py
	cd $(<D); MAKEFILE=1 python $(<F) 2>&1

# Run all tests
tests/test_%.tex: tests/test_%.py
	cd $(<D); python run.py $(<F) 2>&1

foo:
	if [ "${TECTONIC}" = "true" ]; then\
		echo true;\
	else\
		echo false;\
	fi