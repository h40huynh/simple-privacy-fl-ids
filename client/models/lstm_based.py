import pickle
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from opacus import PrivacyEngine

from tqdm import tqdm
import torch
import torch.nn as nn

INPUT_SIZE = 78
HIDDEN_LAYER = 32
OUTPUT_SIZE = 2

DATA_TRAIN_PATH = ""
DATA_TEST_PATH = ""
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CICIDS2017(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_LAYER,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(HIDDEN_LAYER, OUTPUT_SIZE)

    def forward(self, x):
        r, (_, _) = self.lstm(x)
        return self.out(r[:, -1, :])


def load_dataset(n_client, client_id):
    dataset_train_path = "/home/haochu/Desktop/project/datasets/CICIDS2017/custom/dataset_balance_splited_{}_{}.pickle".format(
        n_client, client_id
    )
    dataset_test_path = (
        "/home/haochu/Desktop/project/datasets/CICIDS2017/custom/dataset_test.pickle"
    )
    features_train, labels_train = None, None
    features_test, labels_test = None, None
    with open(dataset_train_path, "rb") as handle:
        features_train, labels_train = pickle.load(handle)
        handle.close()

    with open(dataset_test_path, "rb") as handle:
        features_test, labels_test = pickle.load(handle)
        handle.close()
    return features_train, features_test, labels_train, labels_test


def create_model():
    return Net()


def load_dataloader(features_train, labels_train) -> DataLoader:
    ds = CICIDS2017(features_train, labels_train)
    dl = DataLoader(ds)
    return dl


def load_dataloaders(
    features_train, features_test, labels_train, labels_test
) -> Tuple[DataLoader, DataLoader]:
    traindataset = CICIDS2017(features_train, labels_train)
    testdataset = CICIDS2017(features_test, labels_test)
    trainloader = DataLoader(traindataset, batch_size=64, shuffle=True)
    testloader = DataLoader(testdataset, batch_size=64, shuffle=False)
    return trainloader, testloader


def train(
    net: Net, trainloader: DataLoader, epochs: int, device: torch.device = DEVICE
) -> float:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_list = []

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    net.to(device)
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(
            tqdm(trainloader, desc=f"Epoch {epoch}"), 0
        ):
            # inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.view(-1, 1, INPUT_SIZE).float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 100 == 99:  # print every 100 mini-batches
            #     print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.
        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
        loss_list.append(running_loss / 2000)
        running_loss = 0.0
    return loss_list


def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device = DEVICE,
    is_print=True,
) -> Tuple[float, dict]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    predicted_list = []
    true_label_list = []
    loss = 0.0

    print(f"Testing w/ {len(testloader)} batches")

    # Evaluate the network
    net.to(device)
    net.eval()
    with torch.no_grad():
        for (inputs, labels) in tqdm(testloader, desc="Testing"):
            # inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.view(-1, 1, INPUT_SIZE).float()
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            predicted_list.extend(predicted)
            true_label_list.extend(labels)

    if is_print:
        print(classification_report(true_label_list, predicted_list))
    return (
        loss,
        classification_report(true_label_list, predicted_list, output_dict=True),
    )


# def main():
#     DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("Centralized PyTorch training")
#     print("Load data")
#     trainloader, testloader = load_dataloader()
#     net = Net().to(DEVICE)
#     net.eval()
#     print("Start training")
#     train(net=net, trainloader=trainloader, epochs=2, device=DEVICE)
#     print("Evaluate model")
#     loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
#     print("Loss: ", loss)
#     print("Accuracy: ", accuracy)


# if __name__ == "__main__":
#     main()
