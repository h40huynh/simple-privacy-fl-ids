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
client_id = 0
n_round = 3
n_client = 2
is_leader = False

net, optimizer, criterion = create_model()

# Load dataset
features_train, features_test, labels_train, labels_test = load_dataset(
    n_client, client_id
)

# if not disable_dp:
#     privacy_engine = PrivacyEngine(
#         net,
#         batch_size=64,
#         sample_size=len(features_train),
#         max_grad_norm=1.0,
#         alphas=[10, 100],
#         noise_multiplier=1.3,
#     )
#     privacy_engine.attach(optimizer)
#     print("Training with differential privacy")

testloader = load_dataloader(features_test, labels_test)
dataparts = DatasetSpliter((features_train, labels_train)).split(n_round)
del features_train, labels_train

for round, (features_train, labels_train) in enumerate(dataparts, 1):
    trainloader = load_dataloader(features_train, labels_train)
    train(net, trainloader, epochs, optimizer, criterion)

    if round == n_round:
        test(net, testloader)

