import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lib

transform = lib.transform
class HwdbDataset(Dataset):
    def __init__(self, train=True):
        self.train_data_path = r'D:\Users\pycharm\pythonProject2\data\train'
        self.test_data_path = r'D:\Users\pycharm\pythonProject2\data\test'
        data_path = self.train_data_path if train else self.test_data_path
        class_data_path = []
        for class_name in os.listdir(data_path):
            class_data_path.append(os.path.join(data_path, class_name))
        self.total_file_path = []  #所有文件路径
        for path in tqdm(class_data_path):
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path, i) for i in file_name_list]
            self.total_file_path.extend(file_path_list)

    def __getitem__(self, index):
        file_path = self.total_file_path[index]
        image = Image.open(file_path)
        image = transform(image)
        label = int(file_path.split("\\")[-2])
        return image, label

    def __len__(self):
        return len(self.total_file_path)

def get_dataloader(train=True):
    hwdb_dataset = HwdbDataset(train)
    data_loader = DataLoader(hwdb_dataset, batch_size=lib.batch_size, shuffle=True)
    return data_loader

if __name__ == '__main__':
    for idx, (input, target) in enumerate(get_dataloader()):
        print(idx)
        print(input)
        print(target)
        break
