#!/usr/bin/env bash

#for pid in $(ps -ef | grep "fcc" | awk '{print $2}'); do echo $pid; done
#for pid in $(ps -ef | grep "bcc" | awk '{print $2}'); do kill -9 $pid; done
#

# dropbox
nohup dropbox start &
sleep 120
dropbox status
dropbox throttle 10000 10000
#sudo nethogs

#exit
#nohup ~/anaconda3/bin/python graph/cgcnn/code/hea_script.py --cell bcc > matter_bcc.log &
#nohup ~/anaconda3/bin/python graph/cgcnn/code/hea_script.py --cell fcc > matter_fcc.log &

#dropbox start &
