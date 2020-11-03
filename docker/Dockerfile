FROM continuumio/miniconda3

ENV BK_VERSION=2.2.1
ENV PY_VERSION=3.7.4
ENV SP_BRANCH=master

RUN apt-get update && apt-get -y install git bash libblas-dev g++

RUN conda config --append channels bokeh
RUN conda install --yes --quiet python=${PY_VERSION} pyyaml jinja2 bokeh=${BK_VERSION} numpy scipy theano
RUN conda clean -ay

RUN python -m pip install git+https://github.com/rodluger/starry_process.git@${SP_BRANCH}

RUN git clone --branch $SP_BRANCH https://github.com/rodluger/starry_process.git /starry_process
RUN mv /starry_process/app /. && rm -rf /starry_process

EXPOSE 5006

CMD bokeh serve --allow-websocket-origin="*" app