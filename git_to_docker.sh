#!/bin/bash

# scripts pushes current git repo to origin, 
# then does a docker build and pushes to location specified in build name
# (which can be set using command line parameter)

machine=${1:-"sdp-docker-registry"}

git push origin
sudo docker build --pull -t $machine.kat.ac.za:5000/katsdpcal .
sudo docker push $machine.kat.ac.za:5000/katsdpcal

