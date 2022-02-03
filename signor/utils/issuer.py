# Created at 2020-04-07
# Summary: script for Jie Zhang

""" run shell script to generate data for hea_data """

import os

def banner(text='', ch='=', length=160, compact=False):
    """ http://bit.ly/2vfTDCr
        print a banner
    """
    spaced_text = ' %s ' % text
    banner = spaced_text.center(length, ch)
    print(banner)
    if not compact:
        print()



class issuer():
    def __init__(self, root):
        """ set root directory"""
        self.root = root

    def change_dir(self, hea):
        """
        cd to fcc/bcc of hea, and execute a script
        :param hea:
        :return:
        """

        # fcc
        hea_fcc_dir = f'{self.root}{hea}/fcc/' # change hea fcc dir
        os.chdir(hea_fcc_dir)
        self._exe_cmd('pwd')
        self.exe_script(f = 'abc.sh') # change script name

        # bcc. follow the same format as fcc.
        hea_bcc_dir = f'{self.root}{hea}/bcc/'
        os.chdir(hea_bcc_dir)
        self._exe_cmd('pwd')
        self.exe_script(f = 'abc.sh')

    def exe_script(self, f='abc.sh'):
        # assert os.path.isfile(f), f'File {f} does not exist'

        cmd = f'chmod +x {f} && ./{f}'
        self._exe_cmd(cmd)

    def _exe_cmd(self, cmd):
        banner(f'Executing {cmd}')
        os.system(cmd)
        banner(f'Done Executing {cmd}')

if __name__ == '__main__':
    root = '/home/cai.507/Documents/DeepLearning/Signor/tests/hea/'
    iss = issuer(root)
    for hea in ['hea1', 'hea2', 'hea3']:
        iss.change_dir(hea)