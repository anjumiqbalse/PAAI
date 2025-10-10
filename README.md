<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Enhancing Fast Adversarial Training: Precision-Aware Initialization with Historical Perturbation Guidance </h1>
<p align='left' style="text-align:left;font-size:1.2em;">
</p>

## Introduction

<p align="center"> 

This repository contains the implementation of our method that enhances **Fast Adversarial Training (FAT)** by introducing **Precision-Aware Adversarial Initialization (PAAI)** for robust deep learning models. Our approach significantly improves robustness against strong attacks (e.g., PGD, AutoAttack) while maintaining the computational efficiency of single-step adversarial training.

>**Key Idea**: Instead of using naive random initialization for adversarial perturbations, PAAI generates multiple candidate perturbations using historical guidance and selects the one that maximizes loss—ensuring stronger, more effective adversarial examples from the start.

</p>

## Supported Models

- VGG19
- ResNet18
- PreActResNet18
- WideResNet

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- NumPy

Install dependencies:
```bash
pip install torch torchvision numpy
```

## Usage
## Training
```
python precision_Cifar10.py 
python precision_Cifar100.py 
```
## Testing
```
python3.8 test_cifar10.py 
python3.8 test_cifar100.py 
```


## Citation

This work is currently submitted. If you use this code, please consider citing it once it is published. You may refer to it as:

> Anjum Iqbal, Weiqiang Kong, and Yasir Iqbal. "Enhancing Fast Adversarial Training: Precision-Aware Initialization with Historical Perturbation Guidance" *Submitted to Pattern Analysis and Applications*, 2024.

We will update this README with the official DOI and BibTeX entry upon acceptance
