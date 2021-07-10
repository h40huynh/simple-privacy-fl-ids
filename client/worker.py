import torch
from torch.optim import optimizer
from spliter import DatasetSpliter
from torch.functional import split

# from models.lstm_based import create_model, load_dataloader, load_dataset, test, train
import requests
import argparse
import gc
from utils import Utils
from opacus import PrivacyEngine

gc.enable()

torch.set_num_threads(1)
# print(torch.get_num_threads())


def init_args():
    parser = argparse.ArgumentParser(
        description="Federated learning client with privacy"
    )
    parser.add_argument(
        "-a",
        "--address",
        help="Address of aggregator",
        type=str,
        default="http://127.0.0.1:5000",
    )
    parser.add_argument(
        "--disable-he",
        help="Disable to use homomorphic encryption",
        action="store_true",
    )
    parser.add_argument(
        "--disable-dp",
        help="Disable to training with difference privacy",
        action="store_true",
    )
    parser.add_argument(
        "--local-only", help="Training local only", action="store_true",
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of training epochs", type=int, default=10
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Name of model",
        choices=["lstm", "vgg16", "vgg11", "dnn"],
        type=str,
        default=10,
        required=True,
    )
    return parser.parse_args()


args = init_args()

disable_dp = args.disable_dp
disable_he = args.disable_he
epochs = args.epochs
server_adress = args.address

if args.model == "lstm":
    from models.lstm_based import (
        create_model,
        load_dataloader,
        load_dataset,
        test,
        train,
    )
elif args.model == "vgg16":
    from models.vgg16_based import (
        create_model,
        load_dataloader,
        load_dataset,
        test,
        train,
    )
elif args.model == "vgg11":
    from models.vgg11_based import (
        create_model,
        load_dataloader,
        load_dataset,
        test,
        train,
    )
elif args.model == "dnn":
    from models.dnn_based import (
        create_model,
        load_dataloader,
        load_dataset,
        test,
        train,
    )
else:
    print("Model name not support")
    raise SystemExit

register_path = f"{server_adress}/register"
global_model_path = f"{server_adress}/model/global"
aggregate_model_path = f"{server_adress}/model/aggregation/"

# Init basic infomation
client_id = None
n_round = None
n_client = None
is_leader = False

# Init client session
session = requests.Session()

# Register and get basic infomation: get this client id, number of round, number client
response = session.request("POST", register_path)
response_json = response.json()

if response_json["success"] == False:
    raise SystemExit
else:
    client_id = response_json["data"]["client_id"]
    n_round = response_json["data"]["n_round"]
    n_client = response_json["data"]["n_client"]
    is_leader = response_json["data"]["is_leader"]

net, optimizer, criterion = create_model()
context = Utils.load_key()

from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

# Leader session
if is_leader:
    print("This client set as leader")
    data_to_send = net.state_dict()
    if not disable_he:
        print("Encrypt")
        data_to_send = Utils.encryptialize(context, net.state_dict().copy(), "fedavg")
    data_to_send = Utils.serialize(data_to_send)
    response = session.request(
        "POST",
        global_model_path,
        files={"model": (f"global.weight.pickle", data_to_send, "application/octet")},
    )
    print("Send init weight successfully")
    del data_to_send
    gc.collect()
else:
    print("This client set as workder")
    print("Recieving init weight ...")
    global_model = session.request("GET", global_model_path).content
    print("Recieve init weight successfully")
    global_model = Utils.deserialize(global_model)
    if not disable_he:
        print("Decrypt")
        global_model = Utils.decryptialize(context, global_model)
    net.load_state_dict(global_model)

# Load dataset

features_train, features_test, labels_train, labels_test = load_dataset(
    n_client, client_id
)

if not disable_dp:
    privacy_engine = PrivacyEngine(
        net,
        batch_size=64,
        sample_size=len(features_train),
        max_grad_norm=1.0,
        alphas=[10, 100],
        noise_multiplier=1.3,
    )
    privacy_engine.attach(optimizer)
    print("Training with differential privacy")

testloader = load_dataloader(features_test, labels_test)
dataparts = DatasetSpliter((features_train, labels_train)).split(n_round)
del features_train, labels_train


for round, (features_train, labels_train) in enumerate(dataparts, 1):
    trainloader = load_dataloader(features_train, labels_train)
    train(net, trainloader, epochs, optimizer, criterion)

    if not disable_dp:
        epsilon, best_alpha = privacy_engine.get_privacy_spent()
        print(
            f" (ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}) for α = {best_alpha}"
        )

    # Send model for aggregation
    update_model = net.state_dict()
    if not disable_he:
        update_model = Utils.encryptialize(context, net.state_dict().copy(), "fedavg")
    update_model = Utils.serialize(update_model)
    update_model = {
        "model": (
            f"{round}_{client_id}_{len(labels_train)}.weight.pickle",
            update_model,
            "application/octet",
        )
    }
    update_model = session.request(
        "POST", aggregate_model_path + str(round), files=update_model
    )
    del update_model
    gc.collect()

    update_model = session.request("GET", aggregate_model_path + str(round)).content
    update_model = Utils.deserialize(update_model)
    if not disable_he:
        update_model = Utils.decryptialize(context, update_model)
    net.load_state_dict(update_model)
    del update_model
    gc.collect()

    if is_leader and round == n_round:
        test(net, testloader)

