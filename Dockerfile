ARG KATSDPDOCKERBASE_REGISTRY=harbor.sdp.kat.ac.za/dpp

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-build AS build



# Install python dependencies
COPY  --chown=kat:kat requirements.txt /tmp/install/
RUN chmod -R 777 /tmp/install
RUN uv pip install --no-deps -r /tmp/install/requirements.txt&&\
    uv pip check

# Install the current package
COPY --chown=kat:kat . /tmp/install/katsdpcal

RUN cd /tmp/install/katsdpcal && \
    uv pip install --no-deps . && \
    uv pip check

WORKDIR /tmp

#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-runtime
LABEL maintainer="sdpdev+katsdpcal@ska.ac.za"

COPY --from=build --chown=kat:kat /home/kat/ve3 /home/kat/ve3

# katcp port
EXPOSE 2048
# L0 SPEAD
EXPOSE 7202/udp

# expose volume for saving report etc.
VOLUME ["/var/kat/data"]

# Cal vomits out log files into the current directory, so it needs to be
# somewhere writable.
WORKDIR /tmp
