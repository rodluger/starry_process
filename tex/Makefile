.PHONY: default, clean, deepclean

default:
	python -c "from build import build; build()"

clean:
	python -c "from build import clean; clean()"

deepclean:
	python -c "from build import clean; clean(remove_data=True)"