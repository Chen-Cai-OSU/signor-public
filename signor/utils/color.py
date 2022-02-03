from colorama import init
from termcolor import colored

init()

if __name__ == '__main__':
    print(colored('Hello, World!', 'red'))
    print(f'a{colored("b", "red")}c')
