from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from PIL import Image
import copy
import json
import os

from utils import split_ssl_data
from argument import RandAugment
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, data, targets=None, transform=None, strong_transform=None, numClasses=10, is_ulb=False, onehot=False):
        super(BaseDataset, self).__init__()

        self.data = data
        self.targets = targets
        self.transform = transform
        self.numClasses = numClasses
        self.is_ulb = is_ulb
        self.onehot = onehot

        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3, 5))
        else:
            self.strong_transform = strong_transform

    def __getitem__(self, idx):
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else self.get_onehot(target_)
    
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)

            if not self.is_ulb:
                return idx, img_w, target
            else:
                # return double img_s
                return idx, img_w, self.strong_transform(img), self.strong_transform(img)

    def __len__(self):
        return len(self.data)

    def get_onehot(self, target):
        label_list = np.zeros(self.num_classes, dtype=np.float32)
        label_list[target] = 1.0
        return label_list

class SSLdataset:
    def __init__(self, dataSource="cifar10", data_dir="./data", train=True, numClasses=10):
        self.dataSource = dataSource
        self.data_dir = data_dir
        self.train = train
        self.numClasses = numClasses
        
        self.transform = self.get_transform(mean=[x / 255 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255 for x in [63.0, 62.1, 66.7]],
                                            crop_size=32,
                                            train=train)

    def get_dset(self, is_ulb=False, strong_transform=None, onehot=False):
        """
        return a dataset that has identical functions as BaseDataset
        """
        data, targets = self.get_data()
        num_classes = self.numClasses
        transform = self.transform

        return BaseDataset(data, targets, transform,
                             strong_transform,num_classes, is_ulb, onehot)


    def get_ssl_dset(self, num_labels, index=None, include_lb_in_ulb=True, strong_transform=None, onehot=False):
        """
        num_labels: number of labeled data would be used
        index: if given, labeled data will be aranged as `index`, instead of randomly shuffled
        include_lb_in_ulb: whether include labeled data in the unlabeled data
        
        returns the labeled dataset and unlabeled dataset
        """
        data, targets = self.get_data()
        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(data, targets, num_labels, self.numClasses, index, include_lb_in_ulb)

        # output the distribution of labeled data for remixmatch
        count = [0 for _ in range(self.numClasses)]
        for c in lb_targets:
            count[c] += 1
        dist = np.array(count, dtype=float)
        dist = dist / dist.sum()
        dist = dist.tolist()
        out = {"distribution": dist}
        output_file = r"./data/"
        output_path = output_file +"labels_"+ str(num_labels) + '.json'
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        with open(output_path, 'w') as w:
            json.dump(out, w)
        

        lb_dset = BaseDataset(lb_data, lb_targets, self.transform, None, self.numClasses, False, onehot)
        ulb_dset = BaseDataset(ulb_data, ulb_targets, self.transform, strong_transform, self.numClasses, True, onehot)

        return lb_dset, ulb_dset

    def get_data(self):
        """
        load dataset from torchvision.datasets
        """
        dset = getattr(torchvision.datasets, self.dataSource.upper())
        dset = dset(self.data_dir, train=self.train, download=True)
        data, targets = dset.data, dset.targets
        return data, targets

    def get_transform(self, mean, std, crop_size, train=True):
        if train:
            return transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(crop_size, padding=4, padding_mode='reflect'),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
        else:
            return transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])


def get_dataLoader(dset, batch_size, shuffle, num_workers, pin_memory, drop_last=True):
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)







if __name__ == "__main__":


    """
    Accuracy = 56.4500%
    Accuracy num = 5645
    Accuracy total = 10000
    """