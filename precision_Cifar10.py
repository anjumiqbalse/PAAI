import argparse
import copy
import logging
import os
import time
import math
from tqdm import tqdm
import numpy as np
import torch
from Data_augmentation import Cutout
from CIFAR10_models import *
import torch.nn as nn
from utils import *
from Feature_model.feature_resnet import *
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.utils.data as data
import random

logger = logging.getLogger(__name__)


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

class MultiTemporalBuffer:
    def __init__(self, capacity, gamma=0.1):
        self.buffer = []
        self.losses = []
        self.alignments = []
        self.capacity = capacity
        self.gamma = gamma
        self.training_step = 0
        
    def get_fused_prior(self, device):
        """Compute temporally-weighted fused prior"""
        if len(self.buffer) == 0:
            return None
            
        k = len(self.buffer)
        
        indices = torch.arange(k, dtype=torch.float32, device=device)
        weights = torch.exp(-self.gamma * indices)
        weights = weights / weights.sum()
        
        
        while len(weights.shape) < 4:
            weights = weights.unsqueeze(-1)
        
        # Weighted fusion
        fused = torch.zeros_like(self.buffer[0])
        for i, delta in enumerate(self.buffer):
            fused += weights[i] * delta
        
        return fused
    
    def compute_replacement_probability(self):
        
        p_base = 0.5
        decay_rate = 0.01
        p_t = p_base * math.exp(-decay_rate * self.training_step)
        return max(0.1, p_t)  # Minimum 10% chance
    
    def update(self, new_delta, new_loss, new_alignment=None):
        
        if new_alignment is None:
            new_alignment = 0.0
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(new_delta.detach().clone())
            self.losses.append(new_loss)
            self.alignments.append(new_alignment)
        else:
            
            worst_idx = np.argmin(self.losses)
            
            
            p_replace = self.compute_replacement_probability()
            
           
            if new_loss > self.losses[worst_idx] or random.random() < p_replace:
                self.buffer[worst_idx] = new_delta.detach().clone()
                self.losses[worst_idx] = new_loss
                self.alignments[worst_idx] = new_alignment
        
        self.training_step += 1

class PAAI:
    
    def __init__(self, model, epsilon, args):
        self.model = model
        self.epsilon = epsilon
        self.buffer = MultiTemporalBuffer(capacity=args.buffer_size, gamma=args.gamma)
        
        
        self.num_candidates = args.num_candidates
        self.rho_min = args.rho_min
        self.rho_max = args.rho_max
        self.beta = args.beta
        self.lambda0 = args.lambda0
        self.tau = args.tau
        self.alignment_gamma = args.alignment_gamma
        self.total_iterations = args.epochs * 50000 // args.batch_size
        self.current_iteration = 0
        
    def get_model_output(self, x):
        
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]
        return output
    
    def compute_gradient_alignment(self, x, y, delta):
        
        
        x_clean = x.clone().detach().requires_grad_(True)
        output_clean = self.get_model_output(x_clean)
        loss_clean = F.cross_entropy(output_clean, y)
        grad_clean = torch.autograd.grad(loss_clean, x_clean, retain_graph=False)[0]
        grad_clean_sign = torch.sign(grad_clean)
        
        
        x_adv = (x + delta).clone().detach().requires_grad_(True)
        output_adv = self.get_model_output(x_adv)
        loss_adv = F.cross_entropy(output_adv, y)
        grad_adv = torch.autograd.grad(loss_adv, x_adv, retain_graph=False)[0]
        grad_adv_sign = torch.sign(grad_adv)
        
        
        grad_clean_flat = grad_clean_sign.flatten(1)
        grad_adv_flat = grad_adv_sign.flatten(1)
        
       
        grad_clean_norm = torch.norm(grad_clean_flat, dim=1, keepdim=True)
        grad_adv_norm = torch.norm(grad_adv_flat, dim=1, keepdim=True)
        
        
        mask = (grad_clean_norm > 1e-8) & (grad_adv_norm > 1e-8)
        alignment = torch.zeros(x.size(0), device=x.device)
        
        if mask.any():
            clean_normalized = grad_clean_flat[mask.squeeze()] / grad_clean_norm[mask]
            adv_normalized = grad_adv_flat[mask.squeeze()] / grad_adv_norm[mask]
            alignment[mask.squeeze()] = F.cosine_similarity(clean_normalized, adv_normalized, dim=1)
        
        return alignment.mean().item()
    
    def get_current_lambda(self):
        
        progress = self.current_iteration / max(1, self.total_iterations)
        current_lambda = self.lambda0 * ((1 + progress) ** self.alignment_gamma)
        return current_lambda
    
    def get_current_rho(self):
       
        progress = self.current_iteration / max(1, self.total_iterations)
        rho_current = self.rho_min + (self.rho_max - self.rho_min) * math.exp(-self.beta * progress)
        return rho_current
    
    def generate_candidates(self, x, y, lower_limit, upper_limit):
        
        candidates = []
        candidate_losses = []
        candidate_alignments = []
        
        
        self.model.zero_grad()
        
        
        fused_prior = self.buffer.get_fused_prior(x.device)
        
        
        rho_current = self.get_current_rho()
        current_lambda = self.get_current_lambda()
        
        
        for i in range(self.num_candidates):
            if fused_prior is not None:
                
                eta = torch.zeros_like(x).uniform_(-rho_current, rho_current)
                delta_i = fused_prior + eta
            else:
                
                delta_i = torch.zeros_like(x)
                for j in range(len(self.epsilon)):
                    delta_i[:, j, :, :].uniform_(-self.epsilon[j][0][0].item(), 
                                                self.epsilon[j][0][0].item())
            
            
            delta_i = clamp(delta_i, lower_limit - x, upper_limit - x)
            delta_i = clamp(delta_i, -self.epsilon, self.epsilon)
            
            candidates.append(delta_i)
            
            
            with torch.no_grad():
                output = self.get_model_output(x + delta_i)
                loss_i = F.cross_entropy(output, y).item()
                alignment_i = self.compute_gradient_alignment(x, y, delta_i)
                
            candidate_losses.append(loss_i)
            candidate_alignments.append(alignment_i)
        
        
        candidate_scores = []
        for i in range(len(candidates)):
            # Normalize scores
            if len(candidate_losses) > 1 and (max(candidate_losses) - min(candidate_losses)) > 1e-8:
                norm_loss = (candidate_losses[i] - min(candidate_losses)) / (max(candidate_losses) - min(candidate_losses) + 1e-8)
            else:
                norm_loss = candidate_losses[i]
                
            if len(candidate_alignments) > 1 and (max(candidate_alignments) - min(candidate_alignments)) > 1e-8:
                norm_align = (candidate_alignments[i] - min(candidate_alignments)) / (max(candidate_alignments) - min(candidate_alignments) + 1e-8)
            else:
                norm_align = candidate_alignments[i]
            
            score = norm_loss + current_lambda * norm_align
            candidate_scores.append(score)
        
        best_idx = np.argmax(candidate_scores)
        delta_init = candidates[best_idx]
        
        return delta_init
    
    def constrained_refinement(self, x, y, delta_init, alpha, lower_limit, upper_limit):
        
        delta = delta_init.clone().detach()
        delta.requires_grad_(True)
        
        
        output = self.get_model_output(x + delta)
        loss = F.cross_entropy(output, y)
        
        
        grad = torch.autograd.grad(loss, delta)[0]
        
        delta_temp = delta + alpha * torch.sign(grad)
        delta_temp = clamp(delta_temp, -self.epsilon, self.epsilon)
        delta_temp = clamp(delta_temp, lower_limit - x, upper_limit - x)
        
        
        alignment = self.compute_gradient_alignment(x, y, delta_temp)
        
        if alignment < self.tau:
           
            x_clean = x.clone().detach().requires_grad_(True)
            output_clean = self.get_model_output(x_clean)
            loss_clean = F.cross_entropy(output_clean, y)
            grad_clean = torch.autograd.grad(loss_clean, x_clean)[0]
            
            delta_adv = delta_init + alpha * torch.sign(grad_clean)
            delta_adv = clamp(delta_adv, -self.epsilon, self.epsilon)
            delta_adv = clamp(delta_adv, lower_limit - x, upper_limit - x)
            
           
            alignment = self.compute_gradient_alignment(x, y, delta_adv)
        else:
            delta_adv = delta_temp
        
        return delta_adv.detach(), alignment
    
    def step(self):
        
        self.current_iteration += 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='CIFAR10', type=str)

    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal', 'paai'],
                        help='Perturbation initialization method')
    parser.add_argument('--normal_mean', default=0, type=float, help='normal_mean')
    parser.add_argument('--normal_std', default=1, type=float, help='normal_std')
    parser.add_argument('--out_dir', default='train_fgsm_RS_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--lamda', default=12, type=float, help='Label Smoothing')
    parser.add_argument('--num-candidates', default=5, type=int, help='Number of candidates for PAAI')

    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')

    parser.add_argument('--factor', default=0.7, type=float)
    parser.add_argument('--length', type=int, default=4,
                        help='length of the holes')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--c_num', default=0.125, type=float)
    parser.add_argument('--EMA_value', default=0.60, type=float)
    parser.add_argument('--buffer-size', default=5, type=int, help='Multi-temporal buffer size (K)')
    parser.add_argument('--gamma', default=0.1, type=float, help='Temporal decay rate')
    parser.add_argument('--rho-min', default=0.01, type=float, help='Minimum exploration radius')
    parser.add_argument('--rho-max', default=0.05, type=float, help='Maximum exploration radius')
    parser.add_argument('--beta', default=2.0, type=float, help='Exploration decay rate')
    parser.add_argument('--lambda0', default=0.5, type=float, help='Initial alignment-weight parameter')
    parser.add_argument('--tau', default=0.7, type=float, help='Alignment threshold')
    parser.add_argument('--alignment-gamma', default=0.5, type=float, help='Lambda scheduling parameter')
    
    return parser.parse_args()

args = get_args()

def get_loaders_cutout(dir_, batch_size):
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

    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
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
    one_hot = np.eye(10)[label.cuda().data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(10 - 1))
    return result

def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss

class EMA(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
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

def get_model_by_name(model_name, feature_extractor=True):
    
    if model_name == "VGG":
        return VGG('VGG19')
    elif model_name == "ResNet18":
        return Feature_ResNet18()
    elif model_name == "PreActResNest18":
            return Feature_ResNet18()
    elif model_name == "WideResNet":
        return WideResNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    args = get_args()
    
    
    if args.delta_init == 'paai':
        args.num_candidates = 5
        args.buffer_size = 8
        args.lambda0 = 0.3
        args.tau = 0.6
        args.beta = 1.5
    
    
    lower_limit = ((0.0 - np.array(cifar10_mean)) / np.array(cifar10_std))
    upper_limit = ((1.0 - np.array(cifar10_mean)) / np.array(cifar10_std))
    lower_limit = torch.FloatTensor(lower_limit).view(1, 3, 1, 1).cuda()
    upper_limit = torch.FloatTensor(upper_limit).view(1, 3, 1, 1).cuda()
    
    output_path = os.path.join(args.out_dir, 'Ours')
    output_path = os.path.join(output_path, 'epsilon_' + str(args.epsilon))
    output_path = os.path.join(output_path, 'alpha_' + str(args.alpha))
    output_path = os.path.join(output_path, 'model_' + str(args.model))
    output_path = os.path.join(output_path, 'factor_' + str(args.factor))
    output_path = os.path.join(output_path, 'length_' + str(args.length))
    output_path = os.path.join(output_path, 'EMA_value_' + str(args.EMA_value))
    output_path = os.path.join(output_path, 'lamda_' + str(args.lamda))
    
    if args.delta_init == 'paai':
        output_path = os.path.join(output_path, 'paai_buffer_' + str(args.buffer_size))
        output_path = os.path.join(output_path, 'candidates_' + str(args.num_candidates))
        output_path = os.path.join(output_path, 'lambda_' + str(args.lambda0))
        output_path = os.path.join(output_path, 'tau_' + str(args.tau))

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

    train_loader, test_loader = get_loaders_cutout(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / torch.tensor(cifar10_std).view(1, 3, 1, 1).cuda()
    alpha = (args.alpha / 255.) / torch.tensor(cifar10_std).view(1, 3, 1, 1).cuda()

    print('==> Building model..')
    logger.info('==> Building model..')
    model = get_model_by_name(args.model)
    model = model.cuda()
    model.train()
    
    
    if args.delta_init == 'paai':
        paai = PAAI(model, epsilon, args)
        print(f"PAAI Initialized with buffer_size={args.buffer_size}, candidates={args.num_candidates}")
        print(f"Alignment threshold τ={args.tau}, λ₀={args.lambda0}")
    
    teacher_model = EMA(model)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        
        milestones = [
            lr_steps * 50 // 110,
            lr_steps * 80 // 110,
            lr_steps * 100 // 110
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.5)
    
    # Training
    prev_robust_acc = 0.
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    best_result = 0
    epoch_train_clean_list = []
    epoch_train_pgd_list = []
    epoch_clean_list = []
    epoch_pgd_list = []
    init_loss = []
    init_acc = []
    final_loss = []
    final_acc = []
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Delta Init: {args.delta_init}")
    if args.delta_init == 'paai':
        print(f"  PAAI Candidates: {args.num_candidates}")
        print(f"  Buffer Size: {args.buffer_size}")
        print(f"  Alignment τ: {args.tau}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_loss = 0
        train_acc = 0
        init_train_loss = 0
        init_train_acc = 0
        train_n = 0
        teacher_model.model.eval()
        
        
        if args.delta_init == 'paai':
            epoch_alignments = []
        
        
        epoch_desc = f"Epoch {epoch+1}/{args.epochs}"
        with tqdm(train_loader, desc=epoch_desc, unit="batch", leave=True, ncols=100) as pbar:
            for i, (X, y) in enumerate(pbar):
                X, y = X.cuda(), y.cuda()
                
                
                if args.delta_init == 'paai':
                    
                    pbar.set_postfix({'stage': 'PAAI init'})
                    delta_init = paai.generate_candidates(X, y, lower_limit, upper_limit)
                    
                    
                    delta, alignment = paai.constrained_refinement(X, y, delta_init, alpha, lower_limit, upper_limit)
                    
                    if args.delta_init == 'paai':
                        epoch_alignments.append(alignment)
                        
                    
                    with torch.no_grad():
                        output = paai.get_model_output(X + delta)
                        loss_val = F.cross_entropy(output, y).item()
                    
                    
                    paai.buffer.update(delta, loss_val, alignment)
                    paai.step()
                    
                elif args.delta_init == 'previous':
                    if i == 0:
                        delta = torch.zeros_like(X).cuda()
                    delta = delta[:X.size(0)]
                    
                elif args.delta_init == 'random' and args.num_candidates > 1:
                    candidate_deltas = []
                    candidate_losses = []
                    for _ in range(args.num_candidates):
                        candidate = torch.zeros_like(X).cuda()
                        for j in range(len(epsilon)):
                            candidate[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                        candidate.data = clamp(candidate, lower_limit - X, upper_limit - X)
                        with torch.no_grad():
                            adv_output = model(X + candidate)
                            if isinstance(adv_output, tuple):
                                adv_output = adv_output[0]
                            loss = criterion(adv_output, y)
                        candidate_deltas.append(candidate)
                        candidate_losses.append(loss.item())
                    best_idx = np.argmax(candidate_losses)
                    delta = candidate_deltas[best_idx].clone().detach()
                    
                elif args.delta_init == 'random':
                    delta = torch.zeros_like(X).cuda()
                    for j in range(len(epsilon)):
                        delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    
                elif args.delta_init == 'normal':
                    delta = torch.zeros_like(X).cuda()
                    for j in range(len(epsilon)):
                        delta[:, j, :, :].normal_(args.normal_mean, args.normal_std)
                    delta.data = clamp(delta, -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                else:
                    delta = torch.zeros_like(X).cuda()
                
                
                X_original = X.clone().detach()
                
                
                delta_temp = delta.clone().detach().requires_grad_(True)
                
               
                with torch.enable_grad():
                    adv_output_tuple = model(X + delta_temp)
                    if isinstance(adv_output_tuple, tuple):
                        adv_output = adv_output_tuple[0]
                        ori_fea_output = adv_output_tuple[1]
                    else:
                        adv_output = adv_output_tuple
                        ori_fea_output = adv_output_tuple
                    
                    adv_loss = F.cross_entropy(adv_output, y)
                
                
                grad = torch.autograd.grad(adv_loss, delta_temp)[0]
                
                
                delta = delta_temp + alpha * torch.sign(grad)
                delta = clamp(delta, -epsilon, epsilon)
                delta = clamp(delta, lower_limit - X, upper_limit - X)
                delta = delta.detach()
                
                
                with torch.no_grad():
                    adv_output_tuple_detached = model(X + delta)
                    if isinstance(adv_output_tuple_detached, tuple):
                        adv_output_detached = adv_output_tuple_detached[0].detach()
                        ori_fea_output_detached = adv_output_tuple_detached[1].detach()
                    else:
                        adv_output_detached = adv_output_tuple_detached.detach()
                        ori_fea_output_detached = adv_output_tuple_detached.detach()
                
                
                with torch.no_grad():
                    clean_output_tuple = model(X)
                    if isinstance(clean_output_tuple, tuple):
                        clean_adv_output = clean_output_tuple[0]
                        clean_ori_fea_output = clean_output_tuple[1]
                    else:
                        clean_adv_output = clean_output_tuple
                        clean_ori_fea_output = clean_output_tuple

                init_train_loss += adv_loss.item() * y.size(0)
                init_train_acc += (clean_adv_output.max(1)[1] == y).sum().item()
                
                
                model.zero_grad()
                
                final_output_tuple = model(X + delta)
                if isinstance(final_output_tuple, tuple):
                    ori_output = final_output_tuple[0]
                    fea_output = final_output_tuple[1]
                else:
                    ori_output = final_output_tuple
                    fea_output = final_output_tuple
                
                
                loss_fn = torch.nn.MSELoss(reduction='mean')
                label_smoothing = torch.tensor(_label_smoothing(y, args.factor)).cuda()

                
                label_loss = LabelSmoothLoss(ori_output, label_smoothing.float())
                
                ori_output_softmax = F.softmax(ori_output, dim=1)
                fea_output_softmax = F.softmax(fea_output, dim=1)
                adv_output_softmax_detached = F.softmax(adv_output_detached, dim=1)
                ori_fea_output_softmax_detached = F.softmax(ori_fea_output_detached, dim=1)
                
                
                consistency_loss1 = loss_fn(ori_output_softmax, adv_output_softmax_detached)
                consistency_loss2 = loss_fn(fea_output_softmax, ori_fea_output_softmax_detached)
                
               
                pixel_diff = loss_fn((X + delta).float(), X_original.float()) + args.c_num
                
                
                loss = label_loss + args.lamda * (consistency_loss1 + consistency_loss2) / pixel_diff

                
                opt.zero_grad()
                loss.backward()
                
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                opt.step()
                
                train_loss += loss.item() * y.size(0)
                train_acc += (ori_output.max(1)[1] == y).sum().item()
                adv_acc = (ori_output_softmax.max(1)[1] == y).sum().item()
                clean_acc = (clean_adv_output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
                
                
                if clean_acc > 0 and adv_acc / max(clean_acc, 1e-8) > 0.3:
                    teacher_model.update_params(model)
                    teacher_model.apply_shadow()

                scheduler.step()
                
                
                postfix = {
                    'loss': f'{train_loss/train_n:.3f}',
                    'acc': f'{train_acc/train_n:.2%}',
                    'lr': f'{scheduler.get_last_lr()[0]:.4f}'
                }
                if args.delta_init == 'paai' and epoch_alignments:
                    postfix['align'] = f'{np.mean(epoch_alignments):.3f}'
                pbar.set_postfix(postfix)

        epoch_time = time.time() - epoch_start_time
        
        init_loss.append(init_train_loss / train_n)
        init_acc.append(init_train_acc / train_n)
        final_loss.append(train_loss / train_n)
        final_acc.append(train_acc / train_n)

        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time, lr, train_loss / train_n, train_acc / train_n)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Time: {epoch_time:.1f}s | LR: {lr:.4f}")
        print(f"  Train Loss: {train_loss/train_n:.4f} | Train Acc: {train_acc/train_n:.2%}")
        if args.delta_init == 'paai' and epoch_alignments:
            print(f"  Avg Alignment: {np.mean(epoch_alignments):.3f}")
            logger.info(f"Epoch {epoch} Alignment: {np.mean(epoch_alignments):.3f}")

       
        print(f"\n  Evaluating model...")
        
        
        model_test = get_model_by_name(args.model)
        model_test.load_state_dict(teacher_model.model.state_dict())
        model_test = model_test.cuda()
        model_test.float()
        model_test.eval()
        
        
        epsilon_scalar = args.epsilon / 255.
        
       
        test_desc = "Clean Evaluation"
        test_loss, test_acc = evaluate_standard(test_loader, model_test, test_desc)
        
    
        pgd_desc = "PGD Attack Evaluation"
        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, epsilon_scalar, pgd_desc)
        
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)
        
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        
        print(f"  Test Results:")
        print(f"    Clean - Loss: {test_loss:.4f}, Acc: {test_acc:.2%}")
        print(f"    PGD   - Loss: {pgd_loss:.4f}, Acc: {pgd_acc:.2%}")
        
        if best_result <= pgd_acc:
            best_result = pgd_acc
            torch.save(model_test.state_dict(), os.path.join(output_path, 'best_model.pth'))
            print(f"  ✓ New best PGD accuracy: {pgd_acc:.2%}")

    total_time = time.time() - start_time
    torch.save(model_test.state_dict(), os.path.join(output_path, 'final_model.pth'))
    
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total Training Time: {total_time/60:.1f} minutes")
    print(f"Best PGD Accuracy: {best_result:.2%}")
    print(f"Final Epoch Results:")
    print(f"  Clean Accuracy: {epoch_clean_list[-1]:.2%}")
    print(f"  PGD Accuracy: {epoch_pgd_list[-1]:.2%}")
    
    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)
    print("\nFull epoch clean accuracies:", [f"{acc:.2%}" for acc in epoch_clean_list])
    print("Full epoch PGD accuracies:", [f"{acc:.2%}" for acc in epoch_pgd_list])

def evaluate_standard(test_loader, model, desc="Evaluating"):
    
    model.eval()
    test_loss = 0
    test_acc = 0
    n = 0
    
    with torch.no_grad():
        with tqdm(test_loader, desc=desc, unit="batch", leave=False, ncols=80) as pbar:
            for X, y in pbar:
                X, y = X.cuda(), y.cuda()
                output = model(X)
                if isinstance(output, tuple):
                    output = output[0]
                loss = F.cross_entropy(output, y).item()
                test_loss += loss * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
                
                pbar.set_postfix({
                    'loss': f'{test_loss/n:.3f}',
                    'acc': f'{test_acc/n:.2%}'
                })
    
    return test_loss / n, test_acc / n

def evaluate_pgd(test_loader, model, epsilon, desc="PGD Attack", num_steps=10, num_restarts=1):
    
    model.eval()
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    
   
    epsilon = epsilon / 255.0
    alpha = (2.5 * epsilon) / num_steps
    
   
    mean_tensor = torch.tensor(cifar10_mean).view(1, 3, 1, 1).cuda()
    std_tensor = torch.tensor(cifar10_std).view(1, 3, 1, 1).cuda()
    
    with tqdm(test_loader, desc=desc, unit="batch", leave=False, ncols=80) as pbar:
        for X, y in pbar:
            X, y = X.cuda(), y.cuda()
            
            
            X_unnorm = X * std_tensor + mean_tensor
            
            best_delta = None
            best_loss = -float('inf')
            
            for restart in range(num_restarts):
                delta = torch.zeros_like(X_unnorm).uniform_(-epsilon, epsilon)
                delta = torch.clamp(X_unnorm + delta, 0, 1) - X_unnorm
                
                for attack_step in range(num_steps):
                    delta.requires_grad = True
                    
                    X_perturbed = X_unnorm + delta
                    X_perturbed_norm = (X_perturbed - mean_tensor) / std_tensor
                    
                    output = model(X_perturbed_norm)
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    loss = F.cross_entropy(output, y)
                    
              
                    model.zero_grad()
                    loss.backward()
                    
                    with torch.no_grad():
                        grad = delta.grad.detach()
                        delta = delta + alpha * torch.sign(grad)
                        delta = torch.clamp(delta, -epsilon, epsilon)
                        delta = torch.clamp(X_unnorm + delta, 0, 1) - X_unnorm
                
                
                with torch.no_grad():
                    X_final = X_unnorm + delta
                    X_final_norm = (X_final - mean_tensor) / std_tensor
                    output = model(X_final_norm)
                    if isinstance(output, tuple):
                        output = output[0]
                    loss_val = F.cross_entropy(output, y).item()
                    
                    if loss_val > best_loss:
                        best_loss = loss_val
                        best_delta = delta.clone()
            
          
            with torch.no_grad():
                if best_delta is not None:
                    X_final = X_unnorm + best_delta
                    X_final_norm = (X_final - mean_tensor) / std_tensor
                else:
                    X_final_norm = X
                    
                output = model(X_final_norm)
                if isinstance(output, tuple):
                    output = output[0]
                
                loss = F.cross_entropy(output, y).item()
                pgd_loss += loss * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
            
            pbar.set_postfix({
                'loss': f'{pgd_loss/n:.3f}',
                'acc': f'{pgd_acc/n:.2%}'
            })
    
    return pgd_loss / n, pgd_acc / n
    
if __name__ == "__main__":
    main()
