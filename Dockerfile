FROM sdp-ingest5.kat.ac.za:5000/docker-base:docker-refactor

MAINTAINER Tom Mauch "tmauch@ska.ac.za"

# Install dependencies
COPY requirements.txt /tmp/install/
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -r /tmp/install/requirements.txt

# Install the current package
COPY . /tmp/install/katsdpcal
WORKDIR /tmp/install/katsdpcal
RUN python ./setup.py clean && pip install --no-index .

# expose L1 spead port
EXPOSE 7202
