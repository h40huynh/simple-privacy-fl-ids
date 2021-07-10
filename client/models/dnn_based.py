import numpy as np
import torch
import torch.nn as nn
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
        return self.features[index], self.labels[index]


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(78, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(x)


def load_dataset(n, id):
    features_train, labels_train, features_test, labels_test = None, None, None, None
    with open(
        f"/home/haochu/Desktop/project/datasets/CICIDS2017/custom/trainset_dnn_splited_{n}_{id}.pickle",
        "rb",
    ) as handle:
        features_train, labels_train = pickle.load(handle)
        handle.close()

    with open(
        f"/home/haochu/Desktop/project/datasets/CICIDS2017/custom/testset_dnn.pickle",
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
            outputs = net(inputs.float())
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
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            # if i % 400 == 399:
        print("[%d, %5d] loss: %.5f" % (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

        # end_time = time.time()
        # print("Finish epoch %d with %.5fs" % (epoch + 1, end_time - start_time))
    return loss_list


def create_model():
    net = DNN()
    return (
        net,
        torch.optim.Adam(net.parameters(), lr=8e-4),
        torch.nn.CrossEntropyLoss(),
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    trainloader, testloader = auto_load_data(2, 0)
    net, optimizer, criterion = create_model()
    train(net, trainloader, 10, optimizer, criterion)
    test(net, testloader)
