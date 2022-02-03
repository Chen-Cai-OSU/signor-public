""" file specification """

from signor.ioio.dir import xt_model_dir


class gnn_matter():
    def __init__(self, xt_id, prop, eph, bucket=False, clf=False):
        self.xt_id = xt_id
        self.prop = prop
        self.bucket = bucket
        self.clf = clf
        self.eph = eph

        self.pretrain_epochs = 501
        self.test_epochs = 50
        self.set_dir()
        self.set_crit()
        self.python = '/home/cai.507/anaconda3/envs/pretrain/bin/python'


    def set_dir(self):
        self.dir = f'{xt_model_dir()}mp-ids-{self.xt_id}/{self.prop}/'
        self.fintune_dir = f'result_clf_{self.clf}_bucket_{self.bucket}/'

    def set_crit(self):
        """ set criterion """
        if self.clf:
            self.cri = 'nn.BCEWithLogitsLoss()'
        else:
            self.cri = 'MAE'


    def pretrain_dir(self):
        pass

    def pretrain_res(self):
        f = f'train_loss_bucket_{self.bucket}_epoch_{self.pretrain_epochs}'
        return f

    def finetune_dir(self):
        pass

    def finetune_res(self, seed=0):
        f = f'result_clf_False_bucket_{self.bucket}/'
        f += f'finetune_seed{seed}/'
        f += f'test_pretrain_epochs_{self.pretrain_epochs}_bucket_{self.bucket}_eph_{self.eph}_test_epochs_{self.test_epochs}'
        return f


if __name__ == '__main__':
    xt_id = 3402
    prop = 'band_gap'
    bucket = False
    eph = 21

    ex = gnn_matter(xt_id, prop, eph, bucket=bucket)
    print(ex.pretrain_res())
    print(ex.finetune_res())
