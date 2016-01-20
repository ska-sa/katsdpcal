FROM sdp-ingest5.kat.ac.za:5000/docker-base

MAINTAINER Tom Mauch "tmauch@ska.ac.za"

# Install deb dependencies (need to be root to do this)
USER root
RUN apt-get -y update && apt-get -y install rst2pdf
USER kat

# Install python dependencies
COPY requirements.txt /tmp/install/
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -r /tmp/install/requirements.txt

# Install the current package
COPY . /tmp/install/katsdpcal
WORKDIR /tmp/install/katsdpcal
RUN python ./setup.py clean && pip install --no-index .

WORKDIR /tmp

# expose L1 spead port
EXPOSE 7202
