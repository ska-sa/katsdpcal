FROM sdp-docker-registry.kat.ac.za:5000/docker-base-build as build

MAINTAINER Tom Mauch "tmauch@ska.ac.za"

# Enable Python 2 venv
ENV PATH="$PATH_PYTHON2" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON2"

# Install python dependencies
COPY requirements.txt /tmp/install/
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -r /tmp/install/requirements.txt

# Install the current package
COPY . /tmp/install/katsdpcal
WORKDIR /tmp/install/katsdpcal
RUN python ./setup.py clean && pip install --no-deps . && pip check

WORKDIR /tmp

#######################################################################

FROM sdp-docker-registry.kat.ac.za:5000/docker-base-runtime

COPY --from=build --chown=kat:kat /home/kat/ve /home/kat/ve
ENV PATH="$PATH_PYTHON2" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON2"

# katcp port
EXPOSE 2048
# L0 SPEAD
EXPOSE 7202/udp

# expose volume for saving report etc.
VOLUME ["/var/kat/data"]
