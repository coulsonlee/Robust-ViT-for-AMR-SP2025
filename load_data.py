import os
import argparse
import logging
import copy
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
from dataset.RML_2016_real import build_RML_2016_real,get_noise_data
from dataset.Multi_Files_Dataloader import build_general_loader
import torchvision.transforms as transforms
import warnings
from datetime import datetime
import glob
import math
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True
now = datetime.now()
if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

eps = 0
pert_list=[]

def get_config_for_training(args):
    ret = {
        "Filename": args.dataset_name,
        "Dataset Percentage": args.dataset_size
    }
    return ret



def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)
    args_copy = copy.deepcopy(args)
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}.log'.format(args.user_note)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

##### Your Eval code here  #####
def evaluate(args, model, criterion, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # data = transform(data)
            if len(batch_data) == 2:
                data, target = batch_data
                data, target = data.to(device), target.to(device)
            elif len(batch_data) == 3:
                data, target, power_level = batch_data
                data, target, power_level = data.to(device), target.to(device), power_level.to(device)
            output = model(data)
            test_loss = criterion(output, target).item()
            test_loss += test_loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))

    return correct / float(n)

def set_dataset_name(input_dict):
    dataset_name = ".".join(input_dict["prefix"])
    dataset_name += "_TP" + ".".join(input_dict["tx_pwr"])
    dataset_name += "_D" + ".".join(input_dict["distance"])
    dataset_name += "_SR" + ".".join(input_dict["samp_date"])
    dataset_name += "_I" + ".".join(input_dict["inter"])
    return dataset_name

def select_files_by_keywords(folder_path, keywords):
    selected_files = []

    all_files = glob.glob(os.path.join(folder_path, '*'))

    for file_path in all_files:

        file_name = os.path.basename(file_path)

        if any(keyword in file_name for keyword in keywords[0]):

            if any(keyword in file_name for keyword in keywords[1]):

                if any(keyword in file_name for keyword in keywords[2]):

                    if any(keyword in file_name for keyword in keywords[3]):
                        selected_files.append(file_name)
    return selected_files

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--dataset-size', type=int, default=100, metavar='N',
                        help='input percentage size (default: 100)')
    parser.add_argument('--prefix',type=str,default='/home/lee/Downloads/Dataset_EIB_Outdoor',help="path to your datasets")
    parser.add_argument('--tx_pwr', type=eval, help='dataset tx power (python list)')
    parser.add_argument('--distance', type=eval,help='dataset distance')
    parser.add_argument('--sample_rate', type=eval, help='dataset sample rate')
    parser.add_argument('--inter', type=eval, help='dataset interference')
    parser.add_argument('--dirct_data', type=str, default=None, help='path to your self datasets')

    args = parser.parse_args()

    keywords = []
    keywords.append(args.tx_pwr)
    keywords.append(args.distance)
    keywords.append(args.sample_rate)
    keywords.append(args.inter)

    selected_files = select_files_by_keywords(args.prefix, keywords)

    print_and_log('\n\n')
    print_and_log('=' * 80)
    torch.manual_seed(args.seed)

    dataloader_list=[]
    for i in selected_files:
        path = os.path.join(args.prefix,i)
        try:
            train_loader, test_loader = build_RML_2016_real(path, args.batch_size)
            dataloader_list.append(test_loader)
        except Exception as e:
            raise RuntimeError(f"Failed to build dataloader for {path}: {e}")
    if args.dirct_data is not None:
        dataloader_list = []
        train_loader, test_loader = build_RML_2016_real(args.dirct_data, args.batch_size)
        dataloader_list.append(test_loader)

    
    ####### Your eval code here ########
    for idx,i in enumerate(dataloader_list):
        test_sa = evaluate(args, model, criterion, device, i, is_test_set=True)

if __name__ == '__main__':
    main()
