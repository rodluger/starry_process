FROM continuumio/miniconda3

ENV BK_VERSION=2.2.1
ENV PY_VERSION=3.7.4
ENV SP_COMMIT=aa50317dd747949d00139f1cc5849d7f39412aff
ENV THROTTLE=0.40
ENV YDEG=15
ENV LOAD_TIMEOUT=2

RUN apt-get update && apt-get -y install git bash libblas-dev g++

RUN conda config --append channels bokeh
RUN conda install --yes --quiet python=${PY_VERSION} pyyaml jinja2 bokeh=${BK_VERSION} numpy scipy theano
RUN conda clean -ay

RUN python -m pip install git+https://github.com/rodluger/starry_process.git@${SP_COMMIT}
RUN echo "from starry_process.app import Application; Application(ydeg=${YDEG},throttle_time=${THROTTLE},load_timeout=${LOAD_TIMEOUT})" >> app.py

EXPOSE 5006

CMD bokeh serve \
    --allow-websocket-origin="*" \
    app.py