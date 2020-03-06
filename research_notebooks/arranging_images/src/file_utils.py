import os
from os.path import basename, expanduser, join, splitext


def file_name_from_path(path):
    file_name, _ = splitext(basename(path))
    return file_name


def paths_from_dir(path):
    return [join(path, file_name) for file_name in os.listdir(path)]


def file_names_in_dir(path):
    return set([
        file_name_from_path(path) for path in paths_from_dir(path)
    ])
