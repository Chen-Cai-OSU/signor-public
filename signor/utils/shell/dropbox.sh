#!/usr/bin/env bash

/home/cai.507/anaconda3/envs/sparsifier/bin/python /home/cai.507/Documents/DeepLearning/Signor/signor/monitor/crontab.py

# dropbox
echo "Running dropbox check"
dropbox start
sleep 480
dropbox status
dropbox throttle 10000 10000
echo ''
#sudo nethogs
