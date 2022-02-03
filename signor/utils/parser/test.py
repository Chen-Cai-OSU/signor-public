# Created at 2021-02-28
# Summary:
from signor.utils.parser.argparser import defalut_argparser
parser = defalut_argparser()
parser.add_argument('--other', type=str, default='abc', help='')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
