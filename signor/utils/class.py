import torch
from pprint import pprint

class foo(torch.nn.Module):

    def __init__(self):
        super(foo, self).__init__()
        self.a = 'x'
        self.b = 'y'
        self.set()

    def loop(self):
        self.__dict__['a'] = '-x'
        self.__dict__['b'] = '-y'
        setattr(self, 'c', 3)


    def set(self):
        emb = torch.nn.Embedding(2,2)
        self.emb_0 = emb
        self.emb_1 = emb

        pprint(self.__dict__)

        for i in range(2):
            setattr(self, f'emb{int(i)}', emb)
            # torch.nn.init.xavier_uniform_(self.__dict__[f'emb{int(i)}'].weight.data)
        torch.nn.init.xavier_uniform_(self.emb0.weight.data)
        torch.nn.init.xavier_uniform_(self.emb1.weight.data)

        # self['c'] = '-c'
        # self.c = 3

if __name__ == '__main__':
    f = foo()
    f.loop()
    # print(f.emb)
