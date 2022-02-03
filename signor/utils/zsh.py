""" some common zsh command """

import os
from signor.format.format import delimiter

d = delimiter()

def getPythonProcess():
    python =  '/home/chen/anaconda3/envs/signor/bin/python'
        #'~/anaconda3/bin/envs/signor/python'

    cmd = f' ps aux | grep "{python}"'
    os.system(cmd)
    d.large()

def getSSHProcess():
    cmd = ' ps aux | grep "ssh -N -L"'
    os.system(cmd)
    d.large()

if __name__ == '__main__':
    getPythonProcess()
    getSSHProcess()