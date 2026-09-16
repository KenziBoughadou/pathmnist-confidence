"""Boucle d'apprentissage explicite, commune à toutes les graines du CNN."""

import random

import numpy as np
import torch
from torch import nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def train_epoch(model, loader, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("Perte d'entraînement non finie")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    if count == 0:
        raise ValueError("Partition d'entraînement vide")
    return {"nll": total_loss / count, "accuracy": correct / count}


def predict_logits(model, loader):
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            logits.append(model(images).numpy())
            labels.append(targets.numpy())
    if not logits:
        raise ValueError("Partition d'évaluation vide")
    return np.concatenate(logits), np.concatenate(labels)
