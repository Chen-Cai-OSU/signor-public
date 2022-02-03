# Created at 2020-06-24
# Summary: test gnu parallel for multiple files
import os
file = './num100'
cmd = f'nohup time parallel --jobs 32 < num128 > out1; nohup time parallel --jobs 10 < {file} > out2; nohup time parallel --jobs 16 < num128 > out3 '

print(cmd)
os.system(cmd)