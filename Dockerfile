ARG KATSDPDOCKERBASE_REGISTRY=harbor.sdp.kat.ac.za/dpp

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-build:uvpipjammy AS build

# Enable Python 3 venv
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Install python dependencies
COPY  --chown=kat:kat requirements.txt /tmp/install/
#RUN install_pinned.py -r /tmp/install/requirements.txt
RUN chmod -R 777 /tmp/install
RUN uv pip compile /tmp/install/requirements.txt \
      -o /tmp/install/requirements.lock && \
    uv pip sync /tmp/install/requirements.lock --strict
# Install the current package
COPY --chown=kat:kat . /tmp/install/katsdpcal
#WORKDIR /tmp/install/katsdpcal
#RUN python ./setup.py clean
#RUN rm -rf build/ dist/ *.egg-info
#RUN pip install --no-deps .
#RUN pip check
RUN cd /tmp/install/katsdpcal && \
    #python3 ./setup.py clean   && \
    uv pip install --no-deps . && \
    uv pip check

WORKDIR /tmp

#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-runtime:uvpipjammy
LABEL maintainer="sdpdev+katsdpcal@ska.ac.za"

COPY --from=build --chown=kat:kat /home/kat/ve3 /home/kat/ve3
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# katcp port
EXPOSE 2048
# L0 SPEAD
EXPOSE 7202/udp

# expose volume for saving report etc.
VOLUME ["/var/kat/data"]

# Cal vomits out log files into the current directory, so it needs to be
# somewhere writable.
WORKDIR /tmp
