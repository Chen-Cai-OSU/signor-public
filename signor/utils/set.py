# 07/14/2021
# sets related utils

def Issubset(s1, s2, hard_assert=False):
    s1, s2 = set(s1), set(s2)
    n1, n2 = len(s1), len(s2)
    if n1 > n2:
        return Issubset(s2, s1)

    if s1.issubset(s2):
        return
    else:
        n_bad_items = 0
        for s in s1:
            if s not in s2:
                if hard_assert: raise Exception(f"s1 is not a subset of s2")
                n_bad_items += 1
                if n_bad_items <= 5:
                    print(f'{n_bad_items}-{s} not in set of size {n2}')
        print('...')
        print(f'total {n_bad_items}/{n1} elements are not found in set of size {n2}')

# ve
def plot_venn3(set1, set2, set3, name=('Set1', 'Set2', 'Set3')):
    # Make the diagram
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3
    # Make the diagram
    venn3([set1, set2, set3], name)
    plt.show()

def plot_venn2(set1, set2, name=('Set1', 'Set2')):
    # Make the diagram
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2
    # Make the diagram
    venn2([set1, set2], name)
    plt.show()

if __name__ == '__main__':
    s1 = set(range(1, 5))
    s2 = set(range(2, 6))
    s3 = set(range(3, 7))
    plot_venn2(s1, s2)
    exit()
    s1, s2 = [1,2, -1, -2, -3, -4, -5, -6], list(range(100))
    print(Issubset(s1, s2))