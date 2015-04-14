#!/bin/bash

machine=${1:-"sdp-ingest5"}

git push origin
sudo docker build -t $machine.kat.ac.za:5000/katcal -f Dockerfile.runcal .
sudo docker push $machine.kat.ac.za:5000/katcal

