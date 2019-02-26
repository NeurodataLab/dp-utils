import os


def recursive_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
