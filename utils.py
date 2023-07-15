import os
import re
import errno
import math
import shutil
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from fine2coarse import *

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.mkdir(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class MultiViewDataset(Dataset):
    def __init__(self, data_dir, transform, args):
        self.transform = transform
        self.train_stage = args.train_stage
        self.data_info = self.get_img_info(data_dir)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        path_imgs, label = self.data_info[index]
        imgs = list()
        for path_img in path_imgs:
            img = Image.open(path_img).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
        return imgs, label

#     @staticmethod
    def get_img_info(self, data_dir):
        data_info = list()
        sub_class_names = os.listdir(data_dir)
        class_names = os.listdir(os.path.join(data_dir, sub_class_names[0]))
        
        for label, class_name in enumerate(class_names):  
            img_num = len(os.listdir(os.path.join(data_dir, sub_class_names[0], class_name))) 
            for i in range(img_num):
                img_paths = list()
                for sub_class_name in sub_class_names:
                    img_names = os.listdir(os.path.join(data_dir, sub_class_name, class_name))
                    # img_names.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))
                    img_names.sort(key=lambda x:int(x.split('.')[0]))
                    img_path = os.path.join(data_dir, sub_class_name, class_name, img_names[i])
                    img_paths.append(img_path)
                
                data_info.append((img_paths, label))
        if "train" in data_dir:
            if self.train_stage == 1:
                return data_info[0::2]
            elif self.train_stage == 2:
                return data_info[1::2]
            elif self.train_stage == 3:
                return data_info
            else:
                print("Dataset Error")
        else:
            return data_info

class MultiViewCoarseDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        path_imgs, label = self.data_info[index]
        imgs = list()
        for path_img in path_imgs:
            img = Image.open(path_img).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
        return imgs, label

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        sub_class_names = os.listdir(data_dir)
        class_names = os.listdir(os.path.join(data_dir, sub_class_names[0]))
        
        for label, class_name in enumerate(class_names): 
            coarse_classname = fine_to_coarse[class_name]
            label = coarse_classnames.index(coarse_classname)
            img_num = len(os.listdir(os.path.join(data_dir, sub_class_names[0], class_name))) 
            for i in range(img_num):
                img_paths = list()
                for sub_class_name in sub_class_names:
                    img_names = os.listdir(os.path.join(data_dir, sub_class_name, class_name))
                    img_names.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))
                    img_path = os.path.join(data_dir, sub_class_name, class_name, img_names[i])
                    img_paths.append(img_path)
                
                data_info.append((img_paths, label))
        return data_info

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:1].view(-1).float()

    return correct_k

def get_view(images, action_sequence):
    batch_size = images[0].size(0)
    views = list()
    
    for i in range(batch_size):
        view = images[int(action_sequence[i])][i]
        views.append(view.view(1, view.size(0), view.size(1), view.size(2)))
    
    return torch.cat(views, 0)


def adjust_learning_rate(optimizer, train_configuration, epoch, training_epoch_num, args):
    """Sets the learning rate"""

    if args.fc_lr:
        if args.train_stage == 1:
            backbone_lr = 1e-1 * 0.5 * args.fc_lr * \
                    (1 + math.cos(math.pi * epoch / training_epoch_num))
            fc_lr = 0.5 * args.fc_lr * \
                        (1 + math.cos(math.pi * epoch / training_epoch_num))
        elif args.train_stage == 3:
            backbone_lr = 1e-1 * 0.5 * args.fc_lr * \
                    (1 + math.cos(math.pi * epoch / training_epoch_num))
            fc_lr = 0.5 * args.fc_lr * \
                    (1 + math.cos(math.pi * epoch / training_epoch_num))
    else:
        if args.train_stage == 1:
            backbone_lr = 0.5 * train_configuration['backbone_lr'] * \
                    (1 + math.cos(math.pi * epoch / training_epoch_num))
            fc_lr = 0.5 * train_configuration['fc_stage_1_lr'] * \
                        (1 + math.cos(math.pi * epoch / training_epoch_num))
        elif args.train_stage == 3:
            backbone_lr = 0.5 * train_configuration['backbone_lr'] * \
                    (1 + math.cos(math.pi * epoch / training_epoch_num))
            fc_lr = 0.5 * train_configuration['fc_stage_3_lr'] * \
                    (1 + math.cos(math.pi * epoch / training_epoch_num))
    
    if args.coarse_training:
        backbone_lr = 0.
        
    optimizer.param_groups[0]['lr'] = backbone_lr
    optimizer.param_groups[1]['lr'] = fc_lr

    for param_group in optimizer.param_groups:
        print(param_group['lr'])


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = checkpoint + '/' + filename
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, checkpoint + '/model_best.pth.tar')

def plot(x, data, labels, title):
    plt.figure()
    for i in range(len(data)):
        plt.plot(x[i], data[i], '-', label=labels[i])
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.savefig(title + ".png")

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def modify_state_dict(state_dict):
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict