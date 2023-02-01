import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from custom_transforms import RandomCirclePut, RandomTextPut
from datasets import MyDataset
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow для тензоров"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)
    
label_converter = json.load(open('../data/image_templates.json', 'r'))
label_converter = {v: int(k) for k, v in label_converter.items()}
    
df = pd.read_csv('../data/images.csv')
df['label'] = df['name'].apply(lambda x: label_converter[x])
df['path'] = '../data/resized_images/' + df['filename']

train_transforms = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        RandomTextPut(),
        RandomCirclePut(),
        transforms.RandomRotation((-25, 25)),
        transforms.RandomPerspective(0.2, 0.5),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_transforms = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_df = df.query('isTemplate')
test_df = df.query('~isTemplate')
train_dataset = ConcatDataset(
    [MyDataset(train_df, "train", train_transforms, "path", "label") for _ in range(15)]
)
val_dataset = MyDataset(test_df, "val", test_transforms, "path", "label")


train = DataLoader(train_dataset, batch_size=128, shuffle=True)
val = DataLoader(val_dataset, batch_size=128, shuffle=True)
dataloaders = {"train": train, "val": val}

model = timm.create_model(
    "vit_small_patch32_224_in21k", pretrained=True, num_classes=train_df.name.nunique()
).to("cuda")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10
load_best_weights = True

losses = {'train': [], 'val': []}
best_accuracy = 0.0
for epoch in range(num_epochs):
    tqdm.write(f"Epoch {epoch:03d}")
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0
        running_corrects = 0

        for x_batch, y_batch in tqdm(dataloaders[phase], desc=f"Phase {phase}"):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            if phase == "train":
                optimizer.zero_grad()
                outputs = model(x_batch)
            else:
                with torch.no_grad():
                    outputs = model(x_batch)
            preds = torch.argmax(outputs, -1)
            loss_value = loss(outputs, y_batch)

            if phase == "train":
                loss_value.backward()
                optimizer.step()

            running_loss += loss_value.item()
            running_corrects += int(torch.sum(preds == y_batch.data)) / len(
                y_batch
            )

        epoch_loss = running_loss / len(dataloaders[phase])
        epoch_acc = running_corrects / len(dataloaders[phase])

        losses[phase].append(epoch_loss)

        if phase == "val" and epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            best_weights = model.state_dict()

        tqdm.write(f"\tLoss: {epoch_loss:0.5f}, Accuracy {epoch_acc:0.5f}")
    if scheduler:
        scheduler.step()
    print("-" * 40)
    
if load_best_weights:
    model.load_state_dict(best_weights)

print(f"Best val Acc: {best_accuracy:4f}")
