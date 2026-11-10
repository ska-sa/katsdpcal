ARG KATSDPDOCKERBASE_REGISTRY=harbor.sdp.kat.ac.za/dpp

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-build:production-tag28102026 AS build

# Enable Python 3 venv
ENV PATH="/home/kat/.local/bin:$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Install python dependencies
COPY  --chown=kat:kat requirements.txt /tmp/install/
RUN chmod -R 777 /tmp/install
RUN uv pip compile /tmp/install/requirements.txt \
      -o /tmp/install/requirements.lock && \
    uv pip install --no-deps -r /tmp/install/requirements.lock&&\
    uv pip check

# Install the current package
COPY --chown=kat:kat . /tmp/install/katsdpcal

ENV LD_LIBRARY_PATH=/home/kat/ve3/lib:${LD_LIBRARY_PATH}
RUN uv pip install --force-reinstall "setuptools<81"
RUN cd /tmp/install/katsdpcal && \
    python3 ./setup.py clean   && \
    uv pip install --no-deps . && \
    uv pip check

WORKDIR /tmp

#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-runtime:production-tag28102026
LABEL maintainer="sdpdev+katsdpcal@ska.ac.za"

COPY --from=build --chown=kat:kat /home/kat/ve3 /home/kat/ve3
COPY --from=build --chown=kat:kat /home/kat/.local /home/kat/.local
ENV PATH="/home/kat/.local/bin:$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"
ENV LD_LIBRARY_PATH="/home/kat/ve3/lib:${LD_LIBRARY_PATH:-}"

USER root
# Required by numba set_num_threads - provides Intel TBB threading support
RUN apt-get update && apt-get install -y libtbb2 && apt-get clean
USER kat

# Install TBB Python bindings for numba threading support
RUN uv pip install tbb==2021.6.0

ENV NUMBA_THREADING_LAYER=tbb

# katcp port
EXPOSE 2048
# L0 SPEAD
EXPOSE 7202/udp

# expose volume for saving report etc.
VOLUME ["/var/kat/data"]

# Cal vomits out log files into the current directory, so it needs to be
# somewhere writable.
WORKDIR /tmp
