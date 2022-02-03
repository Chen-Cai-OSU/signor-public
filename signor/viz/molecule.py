from signor.monitor.time import tf

from signor.monitor.probe import summary

import torch

from signor.ioio.dir import find_files, sig_dir
import os
import os
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from textwrap import wrap

import matplotlib.pyplot as plt
from ase import Atom
# from pymatgen.vis.structure_vtk import StructureVis
from ase.visualize.plot import plot_atoms

from signor.graph.cgcnn.utils.property import property_finder
from signor.ioio.dir import curdir, make_dir
from signor.ioio.dir import find_files


class viz_mol():
    def __init__(self, dir=None, ids=['mp-19', 'mp-20'], props=['elasticity.K_VRH']):
        """
        :param dir: cif dir
        :param ids: a list of mp-ids
        """
        self.basedir = curdir()
        self.cif_dir = os.path.join(self.basedir, '..', 'viz', 'cif_test', '') if dir is None else dir
        self.ids = ids
        self.props_var = props

        make_dir(self.cif_dir)
        self.props = dict()
        for prop in props:
            self._set_prop(prop=prop)

    def write_img(self, id):
        """ write POSCAR/CIF into a image dir
        :param id: mp-19

        """
        assert id[:3] == 'mp-'
        if id not in self.ids:
            return
        assert id in self.ids

        filename = self.cif_dir + id + '.cif'
        img_filename = self.cif_dir + id + '.svg'
        title = self.prop2title(id, props=self.props_var)

        write_image_ASE_mpl(filename=filename, img_filename=img_filename, title=title)

    def _set_prop(self, prop='elasticity.K_VRH'):
        PF = property_finder(self.ids)
        PF.find_props()
        sol = PF.find_prop(prop=prop)  # a dict
        self.props[prop] = sol

    def prop2title(self, id, props=['elasticity.K_VRH']):
        """ from prop dict to title
        :param id: mp-19
        :return str: mp-19, K_VRH: 10
        """
        assert isinstance(props, list)
        title = f'{id}: '

        for prop in props:
            assert prop in self.props.keys()
            if id in self.props[prop].keys():
                title += f'{prop}={self.props[prop].get(id, None)} '
        return title


def write_image_ASE_mpl_old(filename='POSCAR', img_filename="image.svg",
                        radius=0.2, rotation=('30x,40y,50z'), title='abc'):
    """ old one from george """
    if not os.path.exists(filename):
        return f'{filename} doesn\'t exist'

    # ASE
    aaa = AseAtomsAdaptor()
    s = Structure.from_file(filename=filename)
    a = aaa.get_atoms(s)

    # matplotlib
    fig, ax = plt.subplots()
    plot_atoms(a, ax, radii=radius, rotation=rotation)
    # fig.savefig(img_filename)
    plt.title("\n".join(wrap(str(title))))
    plt.savefig(img_filename, bbox_inches='tight')




@tf
def write_image_ASE_mpl_new(filename='POSCAR', img_filename="image.png", radius=0.2, rotation=('15x,30y,0z')):
    """ new one from george
        #todo: test
     """

    if os.path.exists(img_filename):
        return

    # ASE
    aaa = AseAtomsAdaptor()
    s = Structure.from_file(filename=filename)
    a = aaa.get_atoms(s)
    # print(a.get_positions())
    # print(a.get_scaled_positions())
    # print(a.get_cell())
    scp = a.get_scaled_positions()
    at_scp = list(zip(a, scp))
    # print(at_scp)
    c0 = a.get_cell()[0]
    c1 = a.get_cell()[1]
    c2 = a.get_cell()[2]

    a_list = []
    for at_scp_i in at_scp:
        at = at_scp_i[0]
        if all(e == 0 for e in at_scp_i[1]):
            p0 = at.position + c0
            p1 = at.position + c1
            p2 = at.position + c2
            p01 = at.position + c0 + c1
            p02 = at.position + c0 + c2
            p12 = at.position + c1 + c2
            p012 = at.position + c0 + c1 + c2
            a_list.append(Atom(at.symbol, p0))
            a_list.append(Atom(at.symbol, p1))
            a_list.append(Atom(at.symbol, p2))
            a_list.append(Atom(at.symbol, p01))
            a_list.append(Atom(at.symbol, p02))
            a_list.append(Atom(at.symbol, p12))
            a_list.append(Atom(at.symbol, p012))
        elif any(e == 0 for e in at_scp_i[1]):
            if list(at_scp_i[1]).count(0) == 1:
                p = list(at_scp_i[1]).index(0)
                pface = at.position + a.get_cell()[p]
                a_list.append(Atom(at.symbol, pface))
            elif list(at_scp_i[1]).count(0) == 2:
                test_list2 = [0]
                res = []
                i = 0
                while (i < len(at_scp_i[1])):
                    if (test_list2.count(at_scp_i[1][i]) > 0):
                        res.append(i)
                    i += 1
                p1 = res[0]
                p2 = res[1]
                p1edge = at.position + a.get_cell()[p1]
                p2edge = at.position + a.get_cell()[p2]
                p12edge = at.position + a.get_cell()[p1] + a.get_cell()[p2]
                a_list.append(Atom(at.symbol, p1edge))
                a_list.append(Atom(at.symbol, p2edge))
                a_list.append(Atom(at.symbol, p12edge))

    for a_ in a_list:
        a.append(a_)

    # matplotlib
    fig, ax = plt.subplots()
    plot_atoms(a, ax, radii=radius, rotation=rotation)
    ax.set_aspect('equal')
    fig.savefig(img_filename)



def get_cif_name(f):
    """ from /home/cai.507/Documents/DeepLearning/material/Wei/data/TianXie/cif/elasticity.K_VRH/123.cif
        to 123
    """
    assert os.path.isfile(f)
    file = f.split('/')[-1]
    assert file[-4:] == '.cif'
    return file[:-4]


if __name__ == '__main__':
    import glob
    from PIL import Image
    from torchvision import transforms

    img_link = f'{sig_dir()}viz/mpl_image.png'
    img = Image.open(img_link)
    print(img)
    trans = transforms.ToPILImage()

    trans1 = transforms.Compose([
        transforms.CenterCrop((480, 480)),
        transforms.ToTensor(),
    ])


    t = trans1(img) # trans(trans1(img))
    t = t[:3, :, :]

    summary(t, 't')
    plt.imshow(trans(trans1(img)))
    plt.show()
    exit()

    print("t is: ", t.size())
    im = transforms.ToPILImage()(t).convert("RGB")
    summary(im, 'im')


    exit()
    dir = '/Users/admin/Documents/osu/Research/Signor/signor/viz/crystals/'
    files = ['MnAlCu2_mp-3574_conventional_standard.cif', 'POSCAR', 'Zn3Cu_mp-972042_conventional_standard.cif', 'structure.cif',
             'MnAlCu2_mp-3574_primitive.cif','POSCAR2']

    for f in files:
        write_image_ASE_mpl_old(f'{dir}{f}', img_filename=f'old_{f}.svg')
        write_image_ASE_mpl_new(f'{dir}{f}', img_filename=f'new_{f}.svg')
    exit()

    PF = property_finder()
    mpids = PF.get_mpids(f='mp-ids-3402.csv')

    vm = viz_mol(ids=mpids, props=['elasticity.K_VRH', 'elasticity.G_VRH'])

    for i, id in enumerate(mpids):
        vm.write_img(id)
        if i % 10 == 9: print(f'Finish {i}-th cif')
    exit()



    write_image_ASE_mpl(filename='structure.cif')

    read_dir = '/home/cai.507/Documents/DeepLearning/material/Wei/data/TianXie/cif/elasticity.K_VRH/'
    read_f = '17229.cif'
    write_dir = '/home/cai.507/Documents/DeepLearning/Signor/data/material/'

    files = find_files(read_dir, suffix='.cif')
    files = files

    for i, file in enumerate(files):
        read_f = file
        write_f = read_f[:-4] + '.svg'
        kwargs = {'filename': read_dir + read_f, 'img_filename': write_dir + write_f}
        write_image_ASE_mpl(**kwargs)
        if i % 10 == 0:
            print(f'Finish {i}-th cif')


    # write_image_ASE_mpl(filename='structure.cif')
