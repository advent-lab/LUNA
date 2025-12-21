#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from argparse import ArgumentParser
from functools import reduce
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix

from dataset import get_preqnt_dataset
from models import QuantumNeqModel

# TODO: Replace default configs with YAML files.
configs = {
    "first": {
        "hidden_layers": [300, 64, 8],
        "input_bitwidth": 1,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 7,
        "hidden_fanin": 6,
        "output_fanin": 7,
        "weight_decay": 0.0,
        "batch_size": 1024,
        "epochs": 20,
        "learning_rate": 1e-4,
        "seed": 109,
        "checkpoint": None,
    },
    "first-light": {
        "hidden_layers": [300, 64, 8],
        "input_bitwidth": 1,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 5,
        "hidden_fanin": 3,
        "output_fanin": 4,
        "weight_decay": 0.0,
        "batch_size": 1024,
        "epochs": 20,
        "learning_rate": 1e-4,
        "seed": 109,
        "checkpoint": None,
    },
    "second": {
        "hidden_layers": [50, 8],
        "input_bitwidth": 1,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 7,
        "hidden_fanin": 6,
        "output_fanin": 8,
        "weight_decay": 0.0,
        "batch_size": 1024,
        "epochs": 20,
        "learning_rate": 1e-4,
        "seed": 109,
        "checkpoint": None,
    },
    "third": {
        "hidden_layers": [500, 64, 8],
        "input_bitwidth": 1,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 3,
        "hidden_fanin": 4,
        "output_fanin": 7,
        "weight_decay": 0.0,
        "batch_size": 1024,
        "epochs": 20,
        "learning_rate": 1e-4,
        "seed": 109,
        "checkpoint": None,
    },
        "fourth": {
        "hidden_layers": [500, 64, 8],
        "input_bitwidth": 1,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 4,
        "hidden_fanin": 4,
        "output_fanin": 4,
        "weight_decay": 0.0,
        "batch_size": 1024,
        "epochs": 20,
        "learning_rate": 1e-4,
        "seed": 109,
        "checkpoint": None,
    },
    
        "fifth": {
        "hidden_layers": [95, 35, 5],
        "input_bitwidth": 1,
        "hidden_bitwidth": 1,
        "output_bitwidth": 2,
        "input_fanin": 6,
        "hidden_fanin": 6,
        "output_fanin": 4,
        "weight_decay": 0.0,
        "batch_size": 1024,
        "epochs": 5,
        "learning_rate": 1e-4,
        "seed": 109,
        "checkpoint": None,
    },
     "fid-opt": {
        "hidden_layers": [145, 40, 15],
        "input_bitwidth": 1,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 7,
        "hidden_fanin": 6,
        "output_fanin": 8,
        "weight_decay": 0.0,
        "batch_size": 512,
        "epochs": 5,
        "learning_rate": 1e-3,
        "seed": 109,
        "checkpoint": None,
    }, 
    "area-opt": {
        "hidden_layers": [25, 5, 5],
        "input_bitwidth": 1,
        "hidden_bitwidth": 1,
        "output_bitwidth": 1,
        "input_fanin": 6,
        "hidden_fanin": 6,
        "output_fanin": 11,
        "weight_decay": 0.0,
        "batch_size": 512,
        "epochs": 5,
        "learning_rate": 1e-3,
        "seed": 109,
        "checkpoint": None,
    },
    "lat-opt": {
        "hidden_layers": [145, 35, 15],
        "input_bitwidth": 1,
        "hidden_bitwidth": 1,
        "output_bitwidth": 2,
        "input_fanin": 6,
        "hidden_fanin": 6,
        "output_fanin": 7,
        "weight_decay": 0.0,
        "batch_size": 512,
        "epochs": 5,
        "learning_rate": 1e-3,
        "seed": 109,
        "checkpoint": None,
    },
    "balanced": {
        "hidden_layers": [130, 40, 10],
        "input_bitwidth": 1,
        "hidden_bitwidth": 1,
        "output_bitwidth": 2,
        "input_fanin": 6,
        "hidden_fanin": 6,
        "output_fanin": 7,
        "weight_decay": 0.0,
        "batch_size": 512,
        "epochs": 5,
        "learning_rate": 1e-3,
        "seed": 109,
        "checkpoint": None,
    }
} 

# A dictionary, so we can set some defaults if necessary
model_config = {
    "hidden_layers": None,
    "input_bitwidth": None,
    "hidden_bitwidth": None,
    "output_bitwidth": None,
    "input_fanin": None,
    "hidden_fanin": None,
    "output_fanin": None,
}

training_config = {
    "weight_decay": None,
    "batch_size": None,
    "epochs": None,
    "learning_rate": None,
    "seed": None,
}

dataset_config = {
    "dataset_path": "",
}

other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
}

def train(model, datasets, train_cfg, options):
    # Create data loaders for training and inference:
    train_loader = DataLoader(datasets["train"], batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(datasets["valid"], batch_size=train_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=train_cfg['batch_size'], shuffle=False)

    # Configure optimizer
    weight_decay = train_cfg["weight_decay"]
    decay_exclusions = ["bn", "bias", "learned_value"] # Make a list of parameters name fragments which will ignore weight decay TODO: make this list part of the train_cfg
    decay_params = []
    no_decay_params = []
    for pname, params in model.named_parameters():
        if params.requires_grad:
            if reduce(lambda a,b: a or b, map(lambda x: x in pname, decay_exclusions)): # check if the current label should be excluded from weight decay
                #print("Disabling weight decay for %s" % (pname))
                no_decay_params.append(params)
            else:
                #print("Enabling weight decay for %s" % (pname))
                decay_params.append(params)
        #else:
            #print("Ignoring %s" % (pname))
    params =    [{'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}]
    optimizer = optim.AdamW(params, lr=train_cfg['learning_rate'], betas=(0.5, 0.999), weight_decay=weight_decay)

    # Configure scheduler
    steps = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=steps*100, T_mult=1)

    # Configure criterion
    criterion = nn.BCEWithLogitsLoss()

    # Push the model to the GPU, if necessary
    if options["cuda"]:
        model.cuda()

    # Setup tensorboard
    writer = SummaryWriter(options["log_dir"])

    # Main training loop
    maxAcc = 0.0
    num_epochs = train_cfg["epochs"]
    for epoch in range(0, num_epochs):
        # Train for this epoch
        model.train()
        accLoss = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if options["cuda"]:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.unsqueeze(1))
            pred = (torch.sigmoid(output.detach()) > 0.75) * 1
            curCorrect = pred.eq(target.unsqueeze(1)).long().sum()
            curAcc = 100.0*curCorrect / len(data)
            correct += curCorrect
            accLoss += loss.detach()*len(data)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (batch_idx%10000 == 0):
            	print(f"Epoch: {epoch}/{num_epochs}\tBatch: {batch_idx}/{len(train_loader)}\tLoss: {loss.detach()}")

            # Log stats to tensorboard
            #writer.add_scalar('train_loss', loss.detach().cpu().numpy(), epoch*steps + batch_idx)
            #writer.add_scalar('train_accuracy', curAcc.detach().cpu().numpy(), epoch*steps + batch_idx)
            #g = optimizer.param_groups[0]
            #writer.add_scalar('LR', g['lr'], epoch*steps + batch_idx)

        accLoss /= len(train_loader.dataset)
        accuracy = 100.0*correct / len(train_loader.dataset)
        print(f"Epoch: {epoch}/{num_epochs}\tTrain Acc (%): {accuracy.detach().cpu().numpy():.4f}\tTrain Loss: {accLoss.detach().cpu().numpy():.3e}")
        #for g in optimizer.param_groups:
        #        print("LR: {:.6f} ".format(g['lr']))
        #        print("LR: {:.6f} ".format(g['weight_decay']))
        writer.add_scalar('avg_train_loss', accLoss.detach().cpu().numpy(), (epoch+1)*steps)
        writer.add_scalar('avg_train_accuracy', accuracy.detach().cpu().numpy(), (epoch+1)*steps)
        val_accuracy = test(model, val_loader, options["cuda"])
        test_accuracy = test(model, test_loader, options["cuda"])
        modelSave = {   'model_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                        'val_accuracy': val_accuracy,
                        'test_accuracy': test_accuracy,
                        'epoch': epoch}
        torch.save(modelSave, options["log_dir"] + "/checkpoint.pth")
        if(maxAcc<val_accuracy):
            torch.save(modelSave, options["log_dir"] + "/best_accuracy.pth")
            maxAcc = val_accuracy
        writer.add_scalar('val_accuracy', val_accuracy, (epoch+1)*steps)
        writer.add_scalar('test_accuracy', test_accuracy, (epoch+1)*steps)
        print(f"Epoch: {epoch}/{num_epochs}\tValid Acc (%): {val_accuracy:.4f}\tTest Acc: {test_accuracy:.4f}")

def test(model, dataset_loader, cuda, disp=0):
    with torch.no_grad():
        model.eval()
        thresh = 0.75
        all_preds = []
        all_targets = []
        correct = 0
        for batch_idx, (data, target) in enumerate(dataset_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = (torch.sigmoid(output.detach()) > thresh) * 1
            curCorrect = pred.eq(target.unsqueeze(1)).long().sum()
            correct += curCorrect
            all_preds.extend(pred.squeeze().tolist())
            all_targets.extend(target.squeeze().tolist())
        accuracy = 100 * float(correct) / len(dataset_loader.dataset)
        
        if disp == 1:
            # Calculate F1 score
            f1 = f1_score(all_targets, all_preds)
            #print("F1 Score:", f1)
            
            # Write confusion matrix to file
            cm = confusion_matrix(all_targets, all_preds)
            #print(cm)
            return accuracy ,f1, cm
            
        return accuracy


# def train_quiet(model, datasets, train_cfg, options):
#     """
#     Train the model quietly (no printing). 
#     Saves checkpoints in options["log_dir"].
#     """
#     train_loader = DataLoader(datasets["train"], batch_size=train_cfg['batch_size'], shuffle=True)
#     val_loader   = DataLoader(datasets["valid"], batch_size=train_cfg['batch_size'], shuffle=False)
#     test_loader  = DataLoader(datasets["test"],  batch_size=train_cfg['batch_size'], shuffle=False)

#     # Optimizer: decay / no-decay groups
#     weight_decay = train_cfg["weight_decay"]
#     decay_exclusions = ["bn", "bias", "learned_value"]
#     decay_params, no_decay_params = [], []
#     for pname, params in model.named_parameters():
#         if params.requires_grad:
#             if any(x in pname for x in decay_exclusions):
#                 no_decay_params.append(params)
#             else:
#                 decay_params.append(params)

#     params = [
#         {"params": decay_params, "weight_decay": weight_decay},
#         {"params": no_decay_params, "weight_decay": 0.0},
#     ]
#     optimizer = optim.AdamW(params, lr=train_cfg['learning_rate'], betas=(0.5, 0.999), weight_decay=weight_decay)

#     steps = len(train_loader)
#     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=steps*100, T_mult=1)

#     criterion = nn.BCEWithLogitsLoss()
#     if options["cuda"]:
#         model.cuda()

#     writer = SummaryWriter(options["log_dir"])

#     maxAcc = 0.0
#     num_epochs = train_cfg["epochs"]

#     for epoch in range(num_epochs):
#         model.train()
#         accLoss, correct = 0.0, 0

#         for data, target in train_loader:
#             if options["cuda"]:
#                 data, target = data.cuda(), target.cuda()
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target.unsqueeze(1))
#             pred = (torch.sigmoid(output.detach()) > 0.75).int()
#             curCorrect = pred.eq(target.unsqueeze(1)).long().sum()
#             correct += curCorrect
#             accLoss += loss.detach() * len(data)
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#         accLoss /= len(train_loader.dataset)
#         accuracy = 100.0 * correct / len(train_loader.dataset)

#         # Save model each epoch
#         val_accuracy  = test_quiet(model, val_loader, options["cuda"])
#         test_accuracy = test_quiet(model, test_loader, options["cuda"])
#         modelSave = {
#             "model_dict": model.state_dict(),
#             "optim_dict": optimizer.state_dict(),
#             "val_accuracy": val_accuracy,
#             "test_accuracy": test_accuracy,
#             "epoch": epoch,
#         }
#         torch.save(modelSave, os.path.join(options["log_dir"], "checkpoint.pth"))
#         if maxAcc < val_accuracy:
#             torch.save(modelSave, os.path.join(options["log_dir"], "best_accuracy.pth"))
#             maxAcc = val_accuracy

#         # minimal logging only to tensorboard
#         writer.add_scalar("avg_train_loss", accLoss.detach().cpu().numpy(), (epoch+1)*steps)
#         writer.add_scalar("avg_train_accuracy", accuracy.detach().cpu().numpy(), (epoch+1)*steps)
#         writer.add_scalar("val_accuracy", val_accuracy, (epoch+1)*steps)
#         writer.add_scalar("test_accuracy", test_accuracy, (epoch+1)*steps)

#     writer.close()


def train_quiet(model, datasets, train_cfg, options):
    """
    Train the model quietly (no printing). 
    Saves checkpoints in options["log_dir"].
    """
    train_loader = DataLoader(datasets["train"], batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader   = DataLoader(datasets["valid"], batch_size=train_cfg['batch_size'], shuffle=False)
    test_loader  = DataLoader(datasets["test"],  batch_size=train_cfg['batch_size'], shuffle=False)
    earlyStop = False

    # Optimizer: decay / no-decay groups
    weight_decay = train_cfg["weight_decay"]
    decay_exclusions = ["bn", "bias", "learned_value"]
    decay_params, no_decay_params = [], []
    for pname, params in model.named_parameters():
        if params.requires_grad:
            if any(x in pname for x in decay_exclusions):
                no_decay_params.append(params)
            else:
                decay_params.append(params)

    params = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(params, lr=train_cfg['learning_rate'], betas=(0.5, 0.999), weight_decay=weight_decay)

    steps = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=steps*100, T_mult=1)

    criterion = nn.BCEWithLogitsLoss()
    if options["cuda"]:
        model.cuda()

    writer = SummaryWriter(options["log_dir"])

    maxAcc = 0.0
    num_epochs = train_cfg["epochs"]

    for epoch in range(num_epochs):
        model.train()
        accLoss, correct = 0.0, 0

        for data, target in train_loader:
            if options["cuda"]:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.unsqueeze(1))
            pred = (torch.sigmoid(output.detach()) > 0.75).int()
            curCorrect = pred.eq(target.unsqueeze(1)).long().sum()
            correct += curCorrect
            accLoss += loss.detach() * len(data)
            loss.backward()
            optimizer.step()
            scheduler.step()

        accLoss /= len(train_loader.dataset)
        accuracy = 100.0 * correct / len(train_loader.dataset)

        # Save model each epoch
        val_accuracy  = test_quiet(model, val_loader, options["cuda"])
        test_accuracy = test_quiet(model, test_loader, options["cuda"])
        modelSave = {
            "model_dict": model.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "epoch": epoch,
        }
        torch.save(modelSave, os.path.join(options["log_dir"], "checkpoint.pth"))
        if maxAcc < val_accuracy:
            torch.save(modelSave, os.path.join(options["log_dir"], "best_accuracy.pth"))
            maxAcc = val_accuracy

        # minimal logging only to tensorboard
        writer.add_scalar("avg_train_loss", accLoss.detach().cpu().numpy(), (epoch+1)*steps)
        writer.add_scalar("avg_train_accuracy", accuracy.detach().cpu().numpy(), (epoch+1)*steps)
        writer.add_scalar("val_accuracy", val_accuracy, (epoch+1)*steps)
        writer.add_scalar("test_accuracy", test_accuracy, (epoch+1)*steps)


    writer.close()
    return earlyStop, model


def test_quiet(model, dataset_loader, cuda):
    """
    Evaluate quietly. 
    Returns fidelity = 1 - 0.5 * (P(0|1) + P(1|0)).
    """
    with torch.no_grad():
        model.eval()
        thresh = 0.75
        all_preds, all_targets = [], []

        for data, target in dataset_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = (torch.sigmoid(output.detach()) > thresh).int()
            all_preds.extend(pred.squeeze().cpu().numpy().tolist())
            all_targets.extend(target.squeeze().cpu().numpy().tolist())

        # Compute confusion matrix manually
        all_preds    = np.array(all_preds)
        all_targets  = np.array(all_targets)
        total_pos    = np.sum(all_targets == 1)
        total_neg    = np.sum(all_targets == 0)
        false_neg    = np.sum((all_targets == 1) & (all_preds == 0))
        false_pos    = np.sum((all_targets == 0) & (all_preds == 1))

        P_0_given_1 = false_neg / total_pos if total_pos > 0 else 0.0
        P_1_given_0 = false_pos / total_neg if total_neg > 0 else 0.0
        fidelity = 1.0 - 0.5 * (P_0_given_1 + P_1_given_0)

        return fidelity



if __name__ == "__main__":
    parser = ArgumentParser(description="LogicNets Network Intrusion Detection Example")
    parser.add_argument('--arch', type=str, choices=configs.keys(), default="nid-s",
        help="Specific the neural network model to use (default: %(default)s)")
    parser.add_argument('--weight-decay', type=float, default=None, metavar='D',
        help="Weight decay (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
        help="Batch size for training (default: %(default)s)")
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
        help="Number of epochs to train (default: %(default)s)")
    parser.add_argument('--learning-rate', type=float, default=None, metavar='LR',
        help="Initial learning rate (default: %(default)s)")
    parser.add_argument('--cuda', action='store_true', default=False,
        help="Train on a GPU (default: %(default)s)")
    parser.add_argument('--seed', type=int, default=None,
        help="Seed to use for RNG (default: %(default)s)")
    parser.add_argument('--input-bitwidth', type=int, default=None,
        help="Bitwidth to use at the input (default: %(default)s)")
    parser.add_argument('--hidden-bitwidth', type=int, default=None,
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)")
    parser.add_argument('--output-bitwidth', type=int, default=None,
        help="Bitwidth to use at the output (default: %(default)s)")
    parser.add_argument('--input-fanin', type=int, default=None,
        help="Fanin to use at the input (default: %(default)s)")
    parser.add_argument('--hidden-fanin', type=int, default=None,
        help="Fanin to use for the hidden layers (default: %(default)s)")
    parser.add_argument('--output-fanin', type=int, default=None,
        help="Fanin to use at the output (default: %(default)s)")
    parser.add_argument('--hidden-layers', nargs='+', type=int, default=None,
        help="A list of hidden layer neuron sizes (default: %(default)s)")
    parser.add_argument('--log-dir', type=str, default='./log',
        help="A location to store the log output of the training run and the output model (default: %(default)s)")
    parser.add_argument('--dataset-path', type=str, default='data',
        help="The file to use as the dataset input (default: %(default)s)")
    parser.add_argument('--checkpoint', type=str, default=None,
        help="Retrain the model from a previous checkpoint (default: %(default)s)")
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options['arch']
    config = {}
    for k in options.keys():
        config[k] = options[k] if options[k] is not None else defaults[k] # Override defaults, if specified.

    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    # Split up configuration options to be more understandable
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    train_cfg = {}
    for k in training_config.keys():
        train_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        options_cfg[k] = config[k]

    # Set random seeds
    random.seed(train_cfg['seed'])
    np.random.seed(train_cfg['seed'])
    torch.manual_seed(train_cfg['seed'])
    os.environ['PYTHONHASHSEED'] = str(train_cfg['seed'])
    if options["cuda"]:
        torch.cuda.manual_seed_all(train_cfg['seed'])
        torch.backends.cudnn.deterministic = True

    # Fetch the datasets
    dataset = {}
    dataset['train'] = get_preqnt_dataset(dataset_cfg['dataset_path'], split="train")
    dataset['valid'] = get_preqnt_dataset(dataset_cfg['dataset_path'], split="val") # This dataset is so small, we'll just use the test set as the validation set, otherwise we may have too few trainings examples to converge.
    dataset['test'] = get_preqnt_dataset(dataset_cfg['dataset_path'], split="test")

    # Instantiate model
    x, y = dataset['train'][0]
    model_cfg['input_length'] = len(x)
    model_cfg['output_length'] = 1
    print(y)
    model = QuantumNeqModel(model_cfg)
    if options_cfg['checkpoint'] is not None:
        print(f"Loading pre-trained checkpoint {options_cfg['checkpoint']}")
        checkpoint = torch.load(options_cfg['checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model_dict'])

    train(model, dataset, train_cfg, options_cfg)


