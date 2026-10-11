import numpy as np
# import scipy.misc as reader
import imageio as reader
import os
from PIL import Image
from torchvision import transforms
# from config import INPUT_SIZE
INPUT_SIZE = (448, 448)

class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        self.preload = False

        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])

        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        
        self.train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        self.test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
        
        if self.preload:
            if self.is_train:
                self.train_img = [reader.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                                self.train_file_list[:data_len]]
                self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            else:
                self.test_img = [reader.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                                self.test_file_list[:data_len]]
                self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            if self.preload:
                img, target = self.train_img[index], self.train_label[index]
            else:
                img_path = os.path.join(self.root, 'images', self.train_file_list[index])
                img, target = reader.imread(img_path), self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            # img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            # img = transforms.RandomCrop(INPUT_SIZE)(img)
            # img = transforms.RandomHorizontalFlip()(img)
            img = transforms.Resize((550, 550))(img)
            img = transforms.RandomCrop(INPUT_SIZE, padding=8)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        else:
            if self.preload:
                img, target = self.test_img[index], self.test_label[index]
            else:
                img_path = os.path.join(self.root, 'images', self.test_file_list[index])
                img, target = reader.imread(img_path), self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            # img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(INPUT_SIZE)(img)
            # img = transforms.ToTensor()(img)
            # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            img = transforms.Resize((550, 550))(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)






