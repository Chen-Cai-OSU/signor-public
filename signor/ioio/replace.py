# https://stackoverflow.com/questions/4205854/python-way-to-recursively-find-and-replace-string-in-text-files
# find
import os, fnmatch
def findReplace(directory, find, replace, filePattern):
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(find, replace)
            with open(filepath, "w") as f:
                f.write(s)

if __name__ == '__main__':
    dir = '/Users/admin/Documents/osu/Research/Signor/signor/ml/Esme/'
    this = 'from Esme'
    that = 'from signor.ml.Esme'
    findReplace(dir, this, that, "*.py")

    # findReplace("some_dir", "find this", "replace with this", "*.txt")
