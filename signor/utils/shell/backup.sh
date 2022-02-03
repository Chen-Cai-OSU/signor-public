#!/usr/bin/env bash

#### github backup (all py files and sh files) ####

backup_time=$(date +'%m/%d/%Y') # https://bit.ly/2NFu06S

#find ./ -name '*.py' -exec du -ch {} + | grep total$ # https://bit.ly/3bEkZU7
#find ./ -name '*.sh' -exec du -ch {} + | grep total$
#find ./ -name '*.yaml' | xargs git add
#
## https://bit.ly/3kewwx5 add all modified files
#git ls-files --modified | xargs git add
#git commit -m "add modified files at ${backup_time}"

find ./ -name '*.sh' | xargs git add
find ./ -name '*.py' | xargs git add
find ./ -name '*.yaml' | xargs git add
git commit -m "backup sh/py/yaml files at ${backup_time}"

rsync -aPz chen@169.228.63.12:/data/chen/learning_to_simulate /data/chen/learning_to_simulate/

exit
# install parallel https://bit.ly/3qS53E9
conda install -c conda-forge parallel

rsync -aPz /home/cai.507/Documents/DeepLearning/GraphCoarsening chen@169.228.63.12:/data/chen/cai.507/cai.507/Documents/DeepLearning/
rsync -aPz /tmp/WaterDrop/  cai.507@CSE-SCSE101549D.cse.ohio-state.edu:/tmp/WaterDrop

# https://bit.ly/3duHey5
#nohup rsync -aPz /home/cai.507/Documents/DeepLearning/Signor/signor chen@169.228.63.12:/home/chen/fromosu/Signor/signor

rsync -avuzh /home/cai.507 chen@169.228.63.12:/home/chen/fromosu/cai.507
rsync -avuzh /home/cai.507/anaconda3/envs/e3nn-test chen@169.228.63.12:/home/chen/fromosu/anaconda3/envs/e3nn-test

# https://bit.ly/3dJwZX5
rsync -avuzh /media/cai.507/Elements/topology-backup/ chen@169.228.63.12:/home/chen/fromosu/backup/

#scp /media/cai.507/Elements/topology-backup/* chen@169.228.63.12:/home/chen/fromosu/backup/
#tar -czf backup-test /home/cai.507/Documents/DeepLearning/Signor/signor/ml | ssh chen@169.228.63.12 "cd /home/chen/ && tar xvzf -"

exit
## test for an existing bus daemon, just to be safe
if test -z "$DBUS_SESSION_BUS_ADDRESS" ; then
    ## if not found, launch a new one
    eval `dbus-launch --sh-syntax --exit-with-session`
fi

# Launch deja-dup
#deja-dup --backup --display=:0
dbus-launch deja-dup --backup


# https://linuxhint.com/inotofy-rsync-bash-live-backups/
# ./liveBackup.bash /home/cai.507/Documents/DeepLearning/Signor /home/cai.507/backup
#while inotifywait -r -e modify,create,delete $1
#
#do
#rsync -avz $1/ $2
#done