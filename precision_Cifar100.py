import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
from Data_augmentation import Cutout
from Cifar100_models import *
from utils import *
from Feature_model.feature_resnet_cifar100 import *
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='CIFAR100', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    parser.add_argument('--normal_mean', default=0, type=float, help='normal_mean')
    parser.add_argument('--normal_std', default=1, type=float, help='normal_std')
    parser.add_argument('--out_dir', default='train_fgsm_RS_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--lamda', default=12, type=float, help='Label Smoothing')
    parser.add_argument('--num-candidates', default=5, type=int, help='Number of candidates for PAAI')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--factor', default=0.5, type=float)
    parser.add_argument('--length', type=int, default=4, help='length of the holes')
    parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
    parser.add_argument('--c_num', default=0.125, type=float)
    parser.add_argument('--EMA_value', default=0.82, type=float)
    return parser.parse_args()

args = get_args()

def get_loaders_cifar100_cutout(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

    num_workers = 0
    train_dataset = datasets.CIFAR100(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader

def _label_smoothing(label, factor):
    one_hot = np.eye(100)[label.cuda().data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(100 - 1))
    return result

def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss

class EMA(object):
    def __init__(self, model, alpha=0.9998, buffer_ema=True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

# Wrapper model to handle tuple outputs during evaluation
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        outputs = self.model(x)
        # If model returns tuple, return only the classification output
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

def main():
    args = get_args()
    output_path = os.path.join(args.out_dir, 'cifar100')
    output_path = os.path.join(output_path, 'factor_' + str(args.factor))
    output_path = os.path.join(output_path, 'EMA_value_' + str(args.EMA_value))
    output_path = os.path.join(output_path, 'lamda_' + str(args.lamda))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logfile = os.path.join(output_path, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(output_path, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders_cifar100_cutout(args.data_dir, args.batch_size)

    # Calculate normalization limits
    mean_tensor = torch.tensor(cifar10_mean).view(1, 3, 1, 1).cuda()
    std_tensor = torch.tensor(cifar10_std).view(1, 3, 1, 1).cuda()
    lower_limit = (0 - mean_tensor) / std_tensor
    upper_limit = (1 - mean_tensor) / std_tensor

    epsilon = (args.epsilon / 255.) / std_tensor
    alpha = (args.alpha / 255.) / std_tensor

    # Build model
    logger.info('==> Building model..')
    if args.model == "VGG":
        model = VGG('VGG19')
    elif args.model == "ResNet18":
        model = Feature_ResNet18()
    elif args.model == "PreActResNest18":
        model = PreActResNet18()
    elif args.model == "WideResNet":
        model = WideResNet()
    model = model.cuda()
    model.train()
    
    teacher_model = EMA(model)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()  # For PAAI candidate selection

    lr_up = 20
    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

    lr_steps = len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0.0, max_lr=args.lr_max,
                                                      step_size_up=lr_steps * lr_up,
                                                      step_size_down=lr_steps * (args.epochs - lr_up))

    # Training
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []
    init_loss = []
    init_acc = []
    final_loss = []
    final_acc = []
    
    for epoch in range(args.epochs):
        epoch_time = 0
        train_loss = 0
        train_acc = 0
        init_train_loss = 0
        init_train_acc = 0
        train_n = 0
        teacher_model.model.eval()
        
        for i, (X, y) in enumerate(train_loader):
            batch_start_time = time.time()
            X, y = X.cuda(), y.cuda()
            
            # Initialize delta based on selected method
            if args.delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
                
            if args.delta_init == 'random' and args.num_candidates > 1:  # PAAI Implementation
                candidate_deltas = []
                candidate_losses = []
                for _ in range(args.num_candidates):
                    candidate = torch.zeros_like(X).cuda()
                    for j in range(len(epsilon)):
                        candidate[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                    candidate.data = clamp(candidate, lower_limit - X, upper_limit - X)
                    with torch.no_grad():
                        adv_output, _ = model(X + candidate)
                        loss = criterion(adv_output, y)
                    candidate_deltas.append(candidate)
                    candidate_losses.append(loss.item())
                best_idx = np.argmax(candidate_losses)
                delta = candidate_deltas[best_idx].clone().detach()
            elif args.delta_init == 'random':  # Standard random initialization
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            elif args.delta_init == 'normal':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].normal_(args.normal_mean, args.normal_std)
                delta.data = clamp(delta, -epsilon, epsilon)
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)

            # Generate adversarial example
            delta.requires_grad = True
            adv_output, ori_fea_output = model(X + delta)
            adv_output = torch.nn.Softmax(dim=1)(adv_output)
            ori_fea_output = torch.nn.Softmax(dim=1)(ori_fea_output)

            loss = F.cross_entropy(adv_output, y)
            init_train_loss += loss.item() * y.size(0)
            init_train_acc += (adv_output.max(1)[1] == y).sum().item()

            loss.backward(retain_graph=True)
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta = delta.detach()
            
            # Calculate final outputs and loss
            ori_output, fea_output = model(X + delta)
            output = torch.nn.Softmax(dim=1)(ori_output)
            fea_output = torch.nn.Softmax(dim=1)(fea_output)

            # Fix loss function to use reduction='mean'
            loss_fn = torch.nn.MSELoss(reduction='mean')
            label_smoothing = torch.tensor(_label_smoothing(y, args.factor)).cuda().float()

            # Calculate distance safely
            dist = loss_fn((X + delta).float(), X.float())
            loss = LabelSmoothLoss(ori_output, label_smoothing) + \
                   args.lamda * (loss_fn(output, adv_output) + loss_fn(fea_output, ori_fea_output)) / \
                   (dist + args.c_num)

            # Optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Update metrics
            train_loss += loss.item() * y.size(0)
            train_acc += (ori_output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            
            # EMA update
            clean_acc = (adv_output.max(1)[1] == y).sum().item()
            adv_acc = (output.max(1)[1] == y).sum().item()
            if clean_acc > 0 and adv_acc / clean_acc < args.EMA_value:
                teacher_model.update_params(model)
                teacher_model.apply_shadow()

            scheduler.step()
            epoch_time += time.time() - batch_start_time

        # Log epoch metrics
        init_loss.append(init_train_loss / train_n)
        init_acc.append(init_train_acc / train_n)
        final_loss.append(train_loss / train_n)
        final_acc.append(train_acc / train_n)

        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time, lr, train_loss / train_n, train_acc / train_n)

        # Evaluate - use wrapper to handle tuple outputs
        model_test = copy.deepcopy(model)
        model_test.load_state_dict(teacher_model.model.state_dict())
        model_test.float().eval()
        wrapped_model = ModelWrapper(model_test)

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, wrapped_model, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, wrapped_model)
        
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)
        
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        
        if best_result <= pgd_acc:
            best_result = pgd_acc
            torch.save(model_test.state_dict(), os.path.join(output_path, 'best_model.pth'))

    # Save final results
    torch.save(model_test.state_dict(), os.path.join(output_path, 'final_model.pth'))
    logger.info("Clean accuracies: %s", str(epoch_clean_list))
    logger.info("PGD accuracies: %s", str(epoch_pgd_list))
    print("Clean accuracies:", epoch_clean_list)
    print("PGD accuracies:", epoch_pgd_list)

if __name__ == "__main__":
    main()