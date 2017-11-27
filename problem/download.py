# -*- coding: utf-8 -*-
import sys
import os

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

DATA_DIR = os.path.join(os.path.expanduser('~'), 'datawarehouse')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)

def get_data_dir():
    return DATA_DIR

# FIXME : Does not change the DATA_DIR global variable !
def set_data_dir(new_data_dir):
    global DATA_DIR
    print('old directory', DATA_DIR)
    if os.path.isdir(new_data_dir):
        print("creating directory :", new_data_dir)
        DATA_DIR = new_data_dir
    print('new directory', DATA_DIR)

def maybe_download(filename, url):
    if not os.path.exists(filename):
        print("downloading " + filename + "...", end='')
        urlretrieve(url, filename)
        print("Done.")
