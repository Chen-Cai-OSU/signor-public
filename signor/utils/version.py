# Created at 2020-04-06
# Summary: get the version of some package

def get_version():
    import torch_geometric
    print(torch_geometric.__version__) # 0.22.1
    print(torch_geometric)

if __name__ == '__main__':
    print('hello')
    get_version()