#!/bin/bash

git push origin
sudo docker build -t sdp-ingest5.kat.ac.za:5000/katcal -f Dockerfile.runcal .
sudo docker push sdp-ingest5.kat.ac.za:5000/katcal

