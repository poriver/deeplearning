import os
import numpy as np
import struct
from PIL import Image
# data文件夹存放转换后的.png文件
data_dir = 'compet_data'
# 路径为存放数据集解压后的.gnt文件
competition_data_dir = os.path.join('', 'competition-gnt')

def read_from_gnt_dir(gnt_dir=competition_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            tagcode = header[5] + (header[4] << 8)
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
            yield image, tagcode

    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode

import pickle

f = open('char_dict', 'rb')
char_dict = pickle.load(f)
f.close()

print(len(char_dict))
print("char_dict=", char_dict)
competition_data_counter = 0

for image, tagcode in read_from_gnt_dir(gnt_dir=competition_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb18030')
    im = Image.fromarray(image)
# 路径为data文件夹下的子文件夹，train为存放训练集.png的文件夹
    dir_name = 'compet_data/' + '%0.5d' % char_dict[tagcode_unicode]
    print(dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    im.convert('RGB').save(dir_name + '/' + str(competition_data_counter) + '.png')
    print("competition_counter=", competition_data_counter)
    competition_data_counter += 1
print('transformation finished ...')





