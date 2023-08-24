import argparse
import os
import shutil

# will set argument
parser = argparse.ArgumentParser(description='split ScanNet')
parser.add_argument('--dir_root', type=str, default='/ssd/ScanNet/')
parser.add_argument('--folder_input', type=str, default='scans')
parser.add_argument('--folder_output', type=str, default='scans_split')
parser.add_argument('--list_train', type=str, default='dataset/scannetv1_train_short.txt')
parser.add_argument('--list_val', type=str, default='dataset/scannetv1_val_short.txt')
parser.add_argument('--list_test', type=str, default='dataset/scannetv1_test_short.txt')

args = parser.parse_args()

dir_root = args.dir_root
folder_input = args.folder_input
folder_output = args.folder_output
list_train = args.list_train
list_val = args.list_val
list_test = args.list_test

dir_input = os.path.join(dir_root, folder_input)
dir_output = os.path.join(dir_root, folder_output)
# Check if the directory exists
if not os.path.exists(dir_output):
    # If it doesn't exist, create it
    os.makedirs(dir_output)

def copy_folder(name_split, list, dir_input, dir_output):
    print('load:', list)
    # create list folder
    dir_train = os.path.join(dir_output, name_split)
    if not os.path.exists(dir_train):
        os.makedirs(os.path.join(dir_train))

    with open(list_train) as f:
        for line in f:
            folder_to_copy = line.strip()
            dir_src = os.path.join(dir_input, folder_to_copy)
            dir_dest = os.path.join(dir_train, folder_to_copy)
            shutil.copytree(dir_src, dir_dest)
            print('copy', dir_src, 'to', dir_dest)

copy_folder('train', list_train, dir_input, dir_output)
copy_folder('val', list_val, dir_input, dir_output)
copy_folder('test', list_test, dir_input, dir_output)

