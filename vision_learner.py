import tqdm
import torch
class VisionModel:
    def __init__(self, model, dataloaders=None, device='cuda', predict_transforms=None, label_encoder=None) -> None:
        self.dataloaders = dataloaders
        self.model = model
        self.device = device
        self.best_weights = model.state_dict()
        self.best_accuracy = 0
        self.is_eval = False
        if not(dataloaders is None):
            self.predict_transforms = dataloaders["val"].dataset.transform
            self.label_encoder = dataloaders["val"].dataset.label_encoder
        elif not(predict_transforms is None or label_encoder is None):
            self.predict_transforms = predict_transforms
            self.label_encoder = label_encoder
        else:
            raise ValueError('dataloaders or predict_transforms and label_encoder not specified')
    def state_dict(self):
        return {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'predict_transforms': self.predict_transforms,
        }
    
    def eval(self, device='cpu'):
        self.model.eval()
        self.to(device)
        self.is_eval = True
        
    def to(self, device):
        self.model.to(device)
        self.device = device
        
    def predict(self, image):
        if self.is_eval == False:
            print('Model state is not eval!!!')
            raise Exception
        image = self.predict_transforms(image)
        with torch.no_grad():
            preds = self.model(torch.unsqueeze(image, 0))
            probs = torch.nn.functional.softmax(preds, dim=1)[0]
            index = preds.argmax()
        return {'pred': self.label_encoder.inverse_transform([index,])[0], 
                'pred_idx': index.item(),
                'probs': probs.tolist(),}
         
    def fit(
        self, num_epochs, criterion, optimizer, scheduler=None, skip_first_val=True, load_best_weights = True,
    ):
        model = self.model
        dataloaders = self.dataloaders
        best_accuracy = 0
        best_weights = self.best_weights
        losses = {
            "train": [],
            "val": [],
        }

        for epoch in range(num_epochs):
            tqdm.write(f"Epoch {epoch:03d}")
            for phase in ["train", "val"]:
                if skip_first_val and epoch == 0 and phase == "val":
                    continue
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
                    loss = criterion(outputs, y_batch)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
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
            self.model.load_state_dict(best_weights)
            self.best_weights = best_weights
            self.best_accuracy = best_accuracy
        print(f"Best val Acc: {best_accuracy:4f}")
        return losses
