import os
import sys
import glob
import argparse

parser = argparse.ArgumentParser(description='generate subdataset')
parser.add_argument('--pile_path', type=str, default='/cache/data/pile_v1', help='path to pile_v1 dataset')
parser.add_argument('--datalist', type=str, default='./datalist/datalist1.txt', help='path to datalist file')
parser.add_argument('--save_path', type=str, default='/cache/data/subdataset/', help='path to save subdataset')
args, _ = parser.parse_known_args()

os.makedirs(args.save_path, exist_ok=True)
dirs = glob.glob(os.path.join(args.pile_path, "*/*.json"))
with open(args.datalist) as datalist:
    lines = datalist.readlines()
for json_file in dirs:
    if os.path.basename(json_file).split(".")[0] + "\n" in lines:
        os.system(f"cp {json_file} {os.path.join(args.save_path, os.path.basename(json_file))}")

    
