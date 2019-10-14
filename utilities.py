import os
from os.path import isfile, join

def correct_zeropadding(input_dir):
    """
    I messed up the zero padding on a long model run so filenames were wrong.
    This script fixes that! 
    """
    # get file list
    file_ls = [f for f in os.listdir(input_dir) if isfile(join(input_dir, f))]
    file_ls = list(filter(lambda x:'.png' in x, file_ls))
    # get new file names for incorrectly labelled
    new_ls = [(s[:9] + '0' + s[9:]) if len(s) == 16 else s for s in file_ls]
    # rename files
    count = 0
    for old, new in zip(file_ls, new_ls):
        if old != new:
            os.rename(input_dir+old, input_dir+new)
            print(old+' renamed to '+new)
            count += 1

    # report
    print('\n'+str(count)+' files renamed..')




