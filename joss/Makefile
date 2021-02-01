# Before running this...
#
#    docker pull openjournals/paperdraft
#

.PHONY: default

FIGURES = $(patsubst %.py,%.pdf,$(sort $(wildcard figures/*.py)))

default: paper.pdf

paper.pdf: paper.md $(FIGURES)
	docker run --rm --volume $(PWD):/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/paperdraft

figures/%.pdf: figures/%.py
	cd $(<D); python $(<F) 2>&1
