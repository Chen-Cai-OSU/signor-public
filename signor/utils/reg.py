# a collection of useful regular expressions
# https://web.mit.edu/hackl/www/lab/turkshop/slides/regex-cheatsheet.pdf

# replace (\cite{}) to \citep{}
# '\(\cite{\b(\w+),(\w+),(\w+)\b}\)'  --> \\citep{\1,\2,\3}