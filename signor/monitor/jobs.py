# Created at 2021-03-03
# Summary: monitor jobs

import os
import os.path as osp
import random

import torch

from signor.configs.gpu import free_gpu
from signor.format.format import red, banner
from signor.ioio.dir import sig_dir
from signor.ioio.file import nol, ftime, lastkline
from signor.ioio.keyboard import yes_or_no
from signor.monitor.time import today, days_before


class job_monitor(object):
    def __init__(self, n_day_before=0):
        # monitor all jobs under self.path
        self.path = f'{sig_dir()}utils/scheduler'
        self.n_fail = 0
        self.n_success = 0
        self.n_day_before = n_day_before
        self.queue = []
        self.cudas = set([f'cuda:{i}' for i in range(torch.cuda.device_count())])


    def find_files(self):
        # find all files to consider
        date = days_before(n=self.n_day_before)
        files = []
        for i in os.listdir(self.path):
            if os.path.isfile(osp.join(self.path, i)) and f'tmp_{date}' in i and 'rerun' not in i:
                files.append(osp.join(self.path, i))
        assert len(files) > 0, f'Find no files containing tmp_{date}'
        return files

    def check_script(self, file, rerun=False):
        # check if all jobs in one script is run successfully
        banner(f'Start checking {file}')
        with open(file) as f:
            content = f.readlines()
        assert content[0] == '#!/usr/bin/env bash\n', 'Not a bash script'

        for cmd in content[1:]:
            line = cmd[:-1]
            assert '>>' in line, f'script {file} do not has >> symbol.'
            wfile = line.split('>>')[-1].strip(' ').split(' ')[0]
            f = wfile # sig_dir() + wfile

            try:
                n_line = nol(f)
                msg = f'{wfile}\nLine #: {n_line}. Modification time: {ftime(f)}\n'
                msg += '--- tail ---\n' + lastkline(f, k=20)
            except FileNotFoundError:
                n_line = -1
                msg = f'{f} not found'

            if n_line < 2 or 'error' in msg:
                print(red(line))
                print(red(msg))
                if rerun and line not in self.queue:
                    self.queue.append(cmd)
                    self.n_fail += 1
            else:
                print(msg)
                self.n_success += 1
            print()
        banner(f'Finish checking {file}')

    def test(self):
        self.files = self.find_files()
        for f in self.files:
            self.check_script(f, rerun=True)
        print(f'n_success jobs: {self.n_success}')
        print(f'n_fail jobs: {self.n_fail}')

    def rerun(self, n_machine=2):
        self.test()
        n_jobs = len(self.queue)
        n_jobs_per_machine = n_jobs // n_machine
        file = f'{sig_dir()}utils/scheduler/tmp_{today()}_rerun.sh'
        yes_or_no(f'Are you going to run the {n_jobs} jobs?\n'
                  f'The cmds is saved at {file} ({n_machine} subfiles)')
        random.shuffle(self.queue)
        gpus = free_gpu(n_machine=n_machine)

        all_scripts = []
        for id in range(n_machine):
            sidx, eidx = id * n_jobs_per_machine, (id + 1) * n_jobs_per_machine
            if id == n_machine - 1: eidx = n_jobs
            tmpfile = file.replace('_rerun', f'_rerun-{id}')
            all_scripts.append(tmpfile)
            with open(tmpfile, 'w') as f:
                f.writelines('#!/usr/bin/env bash\n')
                for cmd in self.queue[sidx: eidx]:
                    if 'cuda:' in cmd:
                        # todo: add assertion
                        _i = cmd.index('cuda:')
                        old = cmd[_i: _i + 6]
                        assert old in self.cudas
                        new = f'cuda:{gpus[id]}'
                        cmd = cmd.replace(old, new)
                    f.writelines(cmd)

        for script in all_scripts:
            print(f'parallel --jobs 2 < {script} & ')


if __name__ == '__main__':
    JM = job_monitor(n_day_before=0)
    # JM.check_script('utils/scheduler/tmp_2021-04-29-22-14.sh')
    JM.test()
