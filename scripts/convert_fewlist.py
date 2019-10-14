import argparse
import random
import os
import numpy as np
from os import path

parser = argparse.ArgumentParser()
parser.add_argument('--droot', type=str, default='/home/bykang/voc')
args = parser.parse_args()

args.droot = args.droot.rstrip('/')
tgt_folder = path.join(args.droot, 'voclist')
src_folder = 'data/vocsplit'

print('===> Converting few-shot name lists.. ')
for name_list in sorted(os.listdir(src_folder)):
    print('  | On ' + name_list)
    # Read from src
    with open(path.join(src_folder, name_list), 'r') as f:
        names = f.readlines()
    
    # Replace data root
    names = [name.replace('/scratch/bykang/datasets', args.droot) 
             for name in names]
    
    with open(path.join(args.droot, 'voclist', name_list), 'w') as f:
        f.writelines(names)

print('===> Converting class to namelist dict file ...')
for fname in ['voc_traindict_full.txt',
              'voc_traindict_bbox_1shot.txt',
              'voc_traindict_bbox_2shot.txt',
              'voc_traindict_bbox_3shot.txt',
              'voc_traindict_bbox_5shot.txt',
              'voc_traindict_bbox_10shot.txt']: 
    full_name = path.join('data', fname)
    print('  | On ' + full_name)
    # Read lines
    with open(full_name, 'r') as f:
        lines = f.readlines()

    # Replace data root
    lines = [line.replace('/scratch/bykang/datasets', args.droot) 
             for line in lines]

    # Rewrite linea
    with open(full_name, 'w') as f:
        f.writelines(lines)

print('===> Finished!')