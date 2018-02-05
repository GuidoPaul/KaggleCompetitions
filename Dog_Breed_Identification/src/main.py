#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import pretrainedmodels

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torch.optim import lr_scheduler

# import torchvision.datasets as datasets
# import torchvision.models as models
import torchvision.transforms as transforms

# import torchvision.utils as utils


class DogsDataset(data.Dataset):
    """Dog breed identification dataset."""

    def __init__(self, img_dir, dataframe, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            dataframe (pandas.core.frame.DataFrame): Pandas dataframe obtained
                by read_csv().
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir,
                                self.labels_frame.id[idx]) + ".jpg"

        image = Image.open(img_name)
        label = self.labels_frame.target[idx]

        if self.transform:
            image = self.transform(image)

        return [image, label]


# Let's visualize a few training images so as to understand the data augmentations.
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def data_loader(all_data_dir, all_labels_df, img_size, batch_size):
    data_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize(img_size),
            # transforms.Resize(int(img_size / 224 * 256)),
            transforms.CenterCrop(img_size),
            # # transforms.RandomResizedCrop(img_size),
            # # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'valid':
        transforms.Compose([
            # Higher scale-up for inception
            transforms.Resize(img_size),
            # transforms.Resize(int(img_size / 224 * 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test':
        transforms.Compose([
            transforms.Resize(img_size),
            # transforms.Resize(int(img_size / 224 * 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    image_datasets = {
        phase: DogsDataset(all_data_dir[phase], all_labels_df[phase],
                           data_transforms[phase])
        for phase in ['train', 'valid', 'test']
    }
    dataloaders = {
        phase: data.DataLoader(
            image_datasets[phase],
            batch_size=batch_size,
            shuffle=False,  # if True, different models have different labels
            num_workers=4)
        for phase in ['train', 'valid', 'test']
    }

    print(
        len(dataloaders['train'].dataset),
        len(dataloaders['valid'].dataset), len(dataloaders['test'].dataset))

    # # # Get a batch of training data
    # inputs, classes = next(iter(dataloaders['train']))

    # # # Make a grid from batch
    # out = utils.make_grid(inputs, nrow=16)
    # imshow(out)
    # plt.show()

    return dataloaders


def save_features_targets(model_name, data_iter):
    # load pretrained model
    if model_name is "nasnetalarge":
        model = pretrainedmodels.__dict__[model_name](
            num_classes=1000, pretrained='imagenet')
    else:
        model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
    model.eval()

    # pretrainedmodels.inceptionresnetv2()

    if use_gpu:
        model.cuda()

    # Don't update non-classifier learned features in the pretrained networks
    for param in model.parameters():
        param.requires_grad = False

    for phase in ['train', 'valid', 'test']:
        features_model_path = "model/features_%s_%s.npy" % (model_name, phase)
        ytargets_model_path = "model/ytargets_%s.npy" % (phase)
        if os.path.exists(features_model_path):
            continue

        features = []
        ytargets = []
        for inputs, labels in data_iter[phase]:
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda(), requires_grad=False)
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(
                    inputs, requires_grad=False), Variable(labels)

            if model_name.startswith('alexnet'):
                x = model.features(inputs)
            elif model_name.startswith('vgg'):
                x = model.make_features(inputs)
            elif model_name.startswith('densenet') or model_name.startswith(
                    'nasnetalarge'):
                x = model.features(inputs)
                x = F.relu(x, inplace=True)
                x = F.avg_pool2d(x, kernel_size=x.size(-1), stride=1)
            else:
                # 'bninception', 'fbresnet152', 'inception', 'resnet'
                # 'resnext101_64x4d', 'inceptionresnetv2'
                x = model.features(inputs)
                # avgpool is important
                x = F.avg_pool2d(x, kernel_size=x.size(-1), stride=1)

                # inception stride=kernel_size
                # x = F.avg_pool2d(x, kernel_size=x.size(-1), stride=1)
            x = x.view(x.size(0), -1)
            features.append(x.data.cpu().numpy())
            ytargets.append(labels.data.cpu().numpy())
        features = np.concatenate(features, axis=0)
        ytargets = np.concatenate(ytargets, axis=0)
        print(features.shape, ytargets.shape)

        np.save(features_model_path, features)
        if os.path.exists(ytargets_model_path):
            continue
        np.save(ytargets_model_path, ytargets)


def load_data(model_names):
    data_iter = {}
    for phase in ['train', 'valid', 'test']:
        if type(model_names) is list:
            features = [
                np.load("model/features_%s_%s.npy" % (model_name, phase))
                for model_name in model_names
            ]
            features = np.concatenate(features, axis=1)
        else:
            features = np.load("model/features_%s_%s.npy" % (model_names,
                                                             phase))
        ytargets = np.load("model/ytargets_%s.npy" % (phase))

        input_dim = features.shape[1]

        features = torch.from_numpy(features)
        ytargets = torch.from_numpy(ytargets)
        tensor_dataset = torch.utils.data.TensorDataset(features, ytargets)
        data_iter[phase] = torch.utils.data.DataLoader(
            tensor_dataset, batch_size=batch_size, shuffle=False)

    return input_dim, data_iter


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    save_filepath = os.path.join("model", "save_" + filename)
    best_filepath = os.path.join("model", "best_" + filename)
    torch.save(state, save_filepath)
    if is_best:
        shutil.copyfile(save_filepath, best_filepath)


def evaluate(model, data_iter_valid, criterion):
    # switch to evaluate mode
    model.train(False)

    running_loss = 0
    running_corrects = 0
    total = 0

    # Iterate over data
    for (inputs, labels) in data_iter_valid:
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # statistics
        total += labels.size(0)
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

    valid_epoch_loss = running_loss / total
    valid_epoch_acc = running_corrects / total

    print("Valid Loss: {:.4f}, Acc: {:.4f}\n".format(valid_epoch_loss,
                                                     valid_epoch_acc))

    return valid_epoch_acc


def train(model, data_iter_train, optimizer, criterion):
    # switch to train mode
    model.train(True)

    running_loss = 0
    running_corrects = 0
    total = 0

    for (inputs, labels) in data_iter_train:
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # statistics
        total += labels.size(0)
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

    train_epoch_loss = running_loss / total
    train_epoch_acc = running_corrects / total

    print("Train Loss: {:.4f}, Acc: {:.4f}".format(train_epoch_loss,
                                                   train_epoch_acc))


def train_model(model_names, input_dim, data_iter, resume=False):
    since = time.time()

    # build model
    model = nn.Sequential(
        nn.Linear(input_dim, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 120), )

    global start_epoch

    if resume:
        # optionally resume from a checkpoint
        if os.path.isfile(resume):
            print("=> Loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            # best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Loaded checkpoint (epoch {})".format(start_epoch))
        else:
            print("=> No checkpoint found at '{}'".format(resume))

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Decay LR by a factor of 0.1 every 30 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=950, gamma=0.9)

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        print("Epoch: {:2d}/{:2d}".format(epoch, num_epochs - 1))

        exp_lr_scheduler.step()

        train(model, data_iter['train'], optimizer, criterion)

        valid_epoch_acc = evaluate(model, data_iter['valid'], criterion)

        if (epoch + 1) % save_period == 0:
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
            }, valid_epoch_acc > best_acc, "ckp_{}.pth.tar".format(epoch + 1))

        # deep copy the model
        if valid_epoch_acc > best_acc:
            best_acc = valid_epoch_acc
            best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best valid Accuracy: {:4f}".format(best_acc))

    # # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


def make_submission(predictions, labels_path, file_path):
    df_pred = pd.read_csv(labels_path)

    for i, c in enumerate(df_pred.columns[1:]):
        df_pred[c] = predictions[:, i]

    df_pred.to_csv(file_path, index=None)


def test(model, model_names, data_iter):
    # switch to evaluate mode
    model.train(False)

    sub_outputs = []
    sub_outputs2 = []

    # Iterate over data
    for inputs, filepath in data_iter['test']:
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        outputs = model(inputs)

        # softmax
        outputs = F.softmax(outputs, dim=1)

        outputs = outputs.data.cpu().numpy()

        outputs2 = np.zeros_like(outputs)
        idxs = outputs.argmax(axis=1)
        for i, idx in enumerate(idxs):
            if (outputs[i][idx] >= 0.9):
                outputs2[i][idx] = 1.0
            else:
                outputs2[i] = outputs[i]

        # sub_outputs.append(outputs.data.cpu().numpy())
        sub_outputs.append(outputs)
        sub_outputs2.append(outputs2)

    sub_outputs = np.concatenate(sub_outputs)
    sub_outputs2 = np.concatenate(sub_outputs2)
    # sub_outputs2 = np.around(sub_outputs2, decimals=4)
    # sub_outputs2 = np.clip(sub_outputs2, clip, 1 - clip)

    make_submission(
        sub_outputs,
        'input/sample_submission.csv',
        'result/pred-20180125-04.csv', )
    make_submission(
        sub_outputs2,
        'input/sample_submission.csv',
        'result/pred-20180125-04-2.csv', )


# data
labels_path = "input/labels.csv"
sample_path = "input/sample_submission.csv"
stanford_lables_path = "input/stanford_lables.csv"
data_dir = "input"
model_dir = "model"
batch_size = 128

# model
lr = 1e-4
start_epoch = 0
num_epochs = 500
save_period = 10
all_model_names = [
    "alexnet",
    "bninception",
    "densenet169",
    "densenet201",
    "fbresnet152",
    "inceptionv3",
    "inceptionv4",
    "inceptionresnetv2",
    "nasnetalarge",
    "resnet18",
    "resnet101",
    "resnet152",
    "resnext101_64x4d",
    "vgg16",
    "vgg19",
    "vgg16_bn",
    "vgg19_bn",
]
select_model_names = [
    "inceptionv3",
    "vgg19_bn",
    "resnext101_64x4d",
    "resnext101_32x4d",
]
select_model_names = [
    "vgg19_bn",
    # "vgg16_bn",
    "resnext101_64x4d",
    # "resnext101_32x4d",
    "inceptionv3",
    "densenet201",
]
# clip = 0.0005

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("use_gpu")

print(pretrainedmodels.model_names)


def main(resume=False):
    # load labels
    labels_df = pd.read_csv(labels_path)
    sample_df = pd.read_csv(sample_path)
    stanford_lables_df = pd.read_csv(stanford_lables_path)

    labelnames = sample_df.keys()[1:]
    codes = range(len(labelnames))
    breed_to_code = dict(zip(labelnames, codes))

    labels_df['target'] = [breed_to_code[x] for x in labels_df.breed]
    stanford_lables_df['target'] = [
        breed_to_code[x] for x in stanford_lables_df.breed
    ]
    sample_df['target'] = [0] * len(sample_df)

    # standard
    # cut = int(len(labels_df) * 0.7)
    # train_df, valid_df = np.split(labels_df, [cut], axis=0)
    # valid_df = valid_df.reset_index(drop=True)

    # all_labels_df = {'train': train_df, 'valid': valid_df, 'test': sample_df}
    # all_data_dir = {
    #     'train': os.path.join(data_dir, 'train'),
    #     'valid': os.path.join(data_dir, 'train'),
    #     'test': os.path.join(data_dir, 'test'),
    # }

    # stanford
    all_labels_df = {
        'train': stanford_lables_df,
        'valid': stanford_lables_df,
        'test': sample_df
    }
    all_data_dir = {
        'train': os.path.join(data_dir, 'Images'),
        'valid': os.path.join(data_dir, 'Images'),
        'test': os.path.join(data_dir, 'test'),
    }

    # Data augmentation and normalization for training
    data_iter_224 = data_loader(
        all_data_dir, all_labels_df, img_size=224, batch_size=batch_size)
    data_iter_299 = data_loader(
        all_data_dir, all_labels_df, img_size=299, batch_size=batch_size)

    # This flag allows you to enable the inbuilt cudnn auto-tuner to find
    # the best algorithm to use for your hardware.
    if use_gpu:
        cudnn.benchmark = True

    for model_name in all_model_names:
        print('Load pretrained {} model on Imagenet'.format(model_name))
        if model_name.startswith('inception'):
            save_features_targets(model_name, data_iter_299)
        else:
            save_features_targets(model_name, data_iter_224)

    # # single feature
    # best_accs = []
    # for model_name in all_model_names:
    #     print("Train {} models".format(model_name))
    #     input_dim, data_iter = load_data(model_name)
    #     model, best_acc = train_model(model_name, input_dim, data_iter)
    #     best_accs.append((model_name, best_acc))
    # df = pd.DataFrame(best_accs, columns=['model', 'best_acc'])
    # df = df.sort_values(by='best_acc', ascending=False)
    # print(df)

    # select feature
    input_dim, data_iter = load_data(select_model_names)
    print("Train {} models".format(select_model_names))
    model, best_acc = train_model(
        select_model_names, input_dim, data_iter, resume=resume)
    print("Test {} models".format(select_model_names))
    test(model, select_model_names, data_iter)


if __name__ == "__main__":
    # main(resume="model/best_ckp_470.pth.tar")
    main()
