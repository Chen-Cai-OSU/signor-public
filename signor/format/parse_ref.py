""" parse bib ref or md """

# @inproceedings{kusano2016persistence,
#   title={Persistence weighted Gaussian kernel for topological data analysis},
#   author={Kusano, Genki and Hiraoka, Yasuaki and Fukumizu, Kenji},
#   booktitle={International Conference on Machine Learning},
#   pages={2004--2013},
#   year={2016}
# }

from pprint import pprint
class bibref():
    def __init__(self):
        pass


if __name__ == '__main__':
    s = "inproceedings{kusano2016persistence,\
  title={Persistence weighted Gaussian kernel for topological data analysis},\
  author={Kusano, Genki and Hiraoka, Yasuaki and Fukumizu, Kenji},\
  booktitle={International Conference on Machine Learning},\
  pages={2004--2013},\
  year={2016}\
    }"

    s = s.split('},')
    new_s = []
    for entry in s:
        while entry[0] == ' ':
            entry = entry[1:]
        while entry[-1] == ' ':
            entry = entry[:-1]
        new_s.append(entry)
    pprint(new_s)
