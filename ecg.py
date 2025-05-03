import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.metrics import roc_auc_score, f1_score
from pyhealth.models import BaseModel

# Define the Prediction Head (shared across all backbones)
class PredictionHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Example Backbone (replace with ResNet1D, LSTM, etc.)
class BackboneWrapper(nn.Module):
    def __init__(self, backbone_model):
        super().__init__()
        self.backbone = backbone_model
        self.output_dim = backbone_model.output_dim
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(self.output_dim, self.output_dim)  # input == output
        )

    def forward(self, x):
        x = self.backbone(x)  # [B, C, T]
        return self.projector(x)  # [B, C]

# Wrapper for backbone + head
class ECGModel(BaseModel):
    def __init__(self, backbone, head, loss_fn, task='multilabel'):
        super().__init__(None, None, None)
        self.model = nn.Sequential(backbone, head)
        self.loss_fn = loss_fn
        self.task = task

    def forward(self, signals):
        logits = self.model(signals)
        return logits

    def compute_metrics(self, logits, labels):
        if self.task == 'multilabel':
            pred = torch.sigmoid(logits).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            return {"roc_auc": roc_auc_score(labels, pred, average="macro")}
        else:
            pred = torch.softmax(logits, dim=1).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            return {"roc_auc": roc_auc_score(labels, pred, average="macro", multi_class="ovr")}
        

class BasicBlock1D(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNet1D(nn.Module):
    def __init__(self, in_channels, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.in_planes = 64
        self.layer0 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(output_dim, 2, stride=2)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock1D(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class NFBlock1D(nn.Module):
    def __init__(self, in_planes, planes, stride=1, alpha=0.2, beta=1.0):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1))
        self.gelu = nn.GELU()
        self.conv2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(planes, planes, kernel_size=3, padding=1))
        self.alpha = alpha
        self.beta = beta
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.utils.parametrizations.weight_norm(nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride))
            )

    def forward(self, x):
        out = self.gelu(self.conv1(x))
        out = self.conv2(out)
        return self.beta * out + self.alpha * self.shortcut(x)

class NFResNet1D(nn.Module):
    def __init__(self, in_channels, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.in_planes = 64
        self.layer0 = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(output_dim, 2, stride=2)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(NFBlock1D(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# Load dataset
def load_data(data_dir):
    X_train = torch.tensor(np.load(os.path.join(data_dir, "X_train.npy"))).float()
    Y_train = torch.tensor(np.load(os.path.join(data_dir, "Y_train.npy"))).float()
    X_val = torch.tensor(np.load(os.path.join(data_dir, "X_val.npy"))).float()
    Y_val = torch.tensor(np.load(os.path.join(data_dir, "Y_val.npy"))).float()
    X_test = torch.tensor(np.load(os.path.join(data_dir, "X_test.npy"))).float()
    Y_test = torch.tensor(np.load(os.path.join(data_dir, "Y_test.npy"))).float()

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# Class weight calculation for class imbalance (multi-class only)
def compute_class_weights(Y):
    class_counts = Y.sum(dim=0)
    weights = 1.0 / (class_counts + 1e-6)
    return weights / weights.sum()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
    
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        probas = torch.sigmoid(logits)
        targets = targets.type(logits.dtype)

        pos_loss = targets * torch.log(probas.clamp(min=self.eps))
        neg_loss = (1 - targets) * torch.log((1 - probas).clamp(min=self.eps))

        if self.clip is not None and self.clip > 0:
            clip_val = torch.clamp(probas, max=self.clip)
            neg_loss *= (1 - clip_val)

        pos_loss *= (1 - probas) ** self.gamma_pos
        neg_loss *= probas ** self.gamma_neg

        loss = -pos_loss - neg_loss
        return loss.mean()


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=250, patience=5, eval_interval=5):
    best_val_score = 0
    best_model = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % eval_interval == 0:
            val_metrics = evaluate_model(model, val_loader)
            val_score = val_metrics["roc_auc"]
            if val_score > best_val_score:
                best_val_score = val_score
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model)
    return model

def evaluate_model(model, data_loader):
    model.eval()
    y_true, y_pred_probs = [], []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            logits = model(x_batch)
            probs = torch.sigmoid(logits)
            y_pred_probs.append(probs.cpu().numpy())
            y_true.append(y_batch.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred_probs = np.vstack(y_pred_probs)
    y_pred_bin = (y_pred_probs >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_pred_probs, average='macro'),
        "f1_score": f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
    }

# Run trainer
def train_and_evaluate(data_dir, backbone_cls, batch_size=128, learning_rate=1e-3, task="multilabel", custom_loss=None):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(data_dir)

    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    test_ds = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    input_channels = X_train.shape[1]
    num_classes = Y_train.shape[1]
    backbone = BackboneWrapper(backbone_cls(input_channels, 256))
    head = PredictionHead(256, num_classes)

    if custom_loss:
        loss_fn = custom_loss
    elif task == "multiclass":
        class_weights = compute_class_weights(Y_train)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    model = ECGModel(backbone, head, loss_fn, task=task)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ##
    metrics = evaluate_model(
        train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=250, patience=5, eval_interval=5),
        test_loader
    )
    print("Test Metrics:", metrics)

# Example usage:
if __name__ == "__main__":
    loss_functions = {
        "BCE": nn.BCEWithLogitsLoss(),
        "Focal": FocalLoss(alpha=1, gamma=2),
        "Asymmetric": AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
    }

    for loss_name, loss_fn in loss_functions.items():
        print(f"\nTraining ResNet1D with {loss_name} Loss")
        train_and_evaluate("data/G12EC/processed", ResNet1D, batch_size=64, learning_rate=1e-3, task="multilabel", custom_loss=loss_fn)

        print(f"\nTraining NFResNet1D with {loss_name} Loss")
        train_and_evaluate("data/G12EC/processed", NFResNet1D, batch_size=64, learning_rate=1e-4, task="multilabel", custom_loss=loss_fn)


