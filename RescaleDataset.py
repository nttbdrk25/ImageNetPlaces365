import os
import shutil
from pathlib import Path
import config_rescale as cf
def read_file(filename=None):
    with open(filename) as f:
        contents = f.readlines() 
    return contents

def rescale_set(path_in_dataset, path_out_dataset, file_name_class):
    list_class = []
    txt_data = open(file_name_class, "r") 
    for ids, txt in enumerate(txt_data):
        s = str(txt.split('\n')[0])
        list_class.append(s)

    for ids, dirs in enumerate(os.listdir(path_in_dataset)):
        for tg_class in list_class:
            if dirs == tg_class:
                print('{} is transferred'.format(dirs))
                shutil.copytree(os.path.join(path_in_dataset,dirs), os.path.join(path_out_dataset,dirs)) 
def main():
    list_files = read_file(cf.path_file_rescale + 'List_file_rescale.txt')
    for file_name_class in list_files:
        #generate_rescale_set(cf.path_in_dataset, cf.path_out_dataset, os.path.join(cf.path_out_rescale,file_name_class))
        print('processing rescale file for train: ' + file_name_class)
        rescale_set(cf.path_in_dataset + 'train', cf.path_out_rescale + Path(file_name_class).stem + '/train', cf.path_file_rescale + file_name_class.rstrip())
        print('processing rescale file for val: ' + file_name_class)
        rescale_set(cf.path_in_dataset + 'val', cf.path_out_rescale + Path(file_name_class).stem + '/val', cf.path_file_rescale + file_name_class.rstrip())
if __name__ == '__main__':
    main()


