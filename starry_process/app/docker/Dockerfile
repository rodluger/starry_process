FROM continuumio/miniconda3

ENV HOME=/tmp
ENV BK_VERSION=2.2.1
ENV PY_VERSION=3.7.4
ENV SP_BRANCH=master
ENV SITE_PACKAGES=/opt/conda/lib/python3.7/site-packages

RUN apt-get update && apt-get -y install git bash libblas-dev g++

RUN conda config --append channels bokeh && \
    conda install --yes --quiet python=${PY_VERSION} pyyaml jinja2 bokeh=${BK_VERSION} numpy scipy theano && \
    conda clean -ay

RUN python -m pip install git+https://github.com/rodluger/starry_process.git@${SP_BRANCH}

EXPOSE 5006

CMD bokeh serve ${SITE_PACKAGES}/starry_process/app