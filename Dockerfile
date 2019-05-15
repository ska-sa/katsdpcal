FROM sdp-docker-registry.kat.ac.za:5000/docker-base-build as build

# Enable Python 3 venv
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Install python dependencies
COPY requirements.txt /tmp/install/
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -r /tmp/install/requirements.txt

# Install the current package
COPY . /tmp/install/katsdpcal
WORKDIR /tmp/install/katsdpcal
RUN python ./setup.py clean
RUN pip install --no-deps .
RUN pip check

WORKDIR /tmp

#######################################################################

FROM sdp-docker-registry.kat.ac.za:5000/docker-base-runtime

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
