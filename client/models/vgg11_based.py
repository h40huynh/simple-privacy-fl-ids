import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import pickle
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from opacus import PrivacyEngine


class CICIDS2017(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        features = np.zeros((81))
        features[:78] = self.features[index]
        features = features.reshape((9, 9, 1))
        features = np.resize(features, (244, 244))
        features = np.stack((features,) * 3, axis=-1)
        return torch.Tensor(features), self.labels[index]


class VGGNetFeatureExtraction(nn.Module):
    def __init__(self):
        super(VGGNetFeatureExtraction, self).__init__()
        vgg = models.vgg11(pretrained=True)
        for p in vgg.parameters():
            p.requires_grad = False

        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.maxpool = nn.MaxPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_dataset(n, id):
    features_train, labels_train, features_test, labels_test = None, None, None, None
    with open(
        f"/home/haochu/Desktop/project/datasets/CICIDS2017/custom/trainset_vgg_splited_{n}_{id}.pickle",
        "rb",
    ) as handle:
        features_train, labels_train = pickle.load(handle)
        handle.close()

    with open(
        f"/home/haochu/Desktop/project/datasets/CICIDS2017/custom/testset_vgg.pickle",
        "rb",
    ) as handle:
        features_test, labels_test = pickle.load(handle)
        handle.close()

    return (
        features_train,
        features_test,
        labels_train,
        labels_test,
    )


def load_dataloader(features, labels):
    ds = CICIDS2017(features, labels)
    return DataLoader(ds, batch_size=64, shuffle=True)


def load_dataloaders(
    features_train, features_test, labels_train, labels_test,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, test, valid"""
    return (
        load_dataloader(features_train, labels_train),
        load_dataloader(features_test, labels_test),
    )


def auto_load_data(n, id):
    (features_train, features_test, labels_train, labels_test,) = load_dataset(n, id)
    return load_dataloaders(features_train, features_test, labels_train, labels_test,)


def test(net: nn.Module, testloader: DataLoader):
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0

    predicted_list = []
    true_labels = []

    net.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(testloader, desc="Testing"), 0):
            outputs = net(inputs.permute(0, 3, 1, 2))
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)

            predicted_list.extend(predicted)
            true_labels.extend(labels)

    print(classification_report(true_labels, predicted_list, digits=4))
    return (
        loss,
        classification_report(true_labels, predicted_list, output_dict=True, digits=4),
    )


def train(net: nn.Module, trainloader: DataLoader, epochs, optimizer, criterion):
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    # optimizer = torch.optim.Adam(net.parameters(), lr=8e-4)
    # criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    loss_list = []
    net.train()
    for epoch in range(epochs):
        # start_time = time.time()
        for i, (inputs, labels) in enumerate(
            tqdm(trainloader, desc=f"Epoch {epoch}"), 0
        ):
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(inputs.permute(0, 3, 1, 2))
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            # if i % 5 == 4:
            #     print("[%d, %5d] loss: %.5f" % (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0

        print("[%d, %5d] loss: %.5f" % (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0
        # end_time = time.time()
        # print("Finish epoch %d with %.5fs" % (epoch + 1, end_time - start_time))
    return loss_list


def create_model() -> nn.Module:
    net = VGGNetFeatureExtraction()
    return (
        net,
        torch.optim.Adam(net.parameters(), lr=8e-4),
        torch.nn.CrossEntropyLoss(),
    )


if __name__ == "__main__":
    trainloader, testloader = auto_load_data(2, 0)
    model = create_model()
    train(model, trainloader, 5)
    test(model, testloader)
