#!/usr/bin/env bash

# https://bit.ly/2Jp3P2w
sudo apt-get install ncdu
ncdu

cd
du -sh ./* | sort -h | tail

#17G     ./anaconda2
#130G    ./anaconda3
#178G    ./Dropbox
#540G    ./Documents
