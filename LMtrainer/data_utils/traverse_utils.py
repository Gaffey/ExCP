import pathlib
import os


def traverse_cache_root(root_dir):
    """
    Find all cached directory in root_dir
    Return all cached directory as a list
    """
    root_path = pathlib.Path(root_dir)
    files = root_path.glob('**/dataset_dict.json')
    cache_dir_list = []
    for f in files:
        subdir, _ = os.path.split(f)
        cache_dir_list.append(subdir)
    return cache_dir_list


def travese_json_root(text_dir):
    """
        Find all json files in text_dir
        Return all cjson files path as a list
    """
    root_path = pathlib.Path(text_dir)
    files = root_path.glob('**/*.json')
    file_list = [str(f) for f in files]
    return file_list

