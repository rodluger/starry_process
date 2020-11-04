.PHONY : build, run, push

build:
	docker build --tag rodluger/starry_process .

run:
	docker run --rm -p 5006:5006 -it rodluger/starry_process

push:
	docker push rodluger/starry_process