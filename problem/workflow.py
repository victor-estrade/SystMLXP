# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


import os
import sys
import datetime

try:
	import builtins as __builtin__
except ImportError as e:
	import  __builtin__


def print(*args, sep=' ', flush=True, end='\n', file=sys.stdout):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    __builtin__.print(now, *args, sep=sep, flush=flush, end=end, file=file)


def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path


def _get_save_directory():
    titanic = '/data/titanic_3/users/vestrade/savings'
    laptop = '/home/estrade/Bureau/PhD/SystML/savings/mnist'
    if os.path.isdir(titanic):
        return titanic
    elif os.path.isdir(laptop):
        return laptop
    else:
        raise FileNotFoundError('savings directory not found.')


