from typing import Dict, List, Tuple
from flask import Flask, request, make_response, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from sql_models import db, TrainProfile
from os import path, remove, rename, mkdir
from glob import glob
from time import sleep
from utils import Utils

import re
import gc
import mmap
import sys
import argparse


def init_args():
    parser = argparse.ArgumentParser(
        description="Federated learning server with privacy"
    )
    parser.add_argument(
        "-p", "--port", help="Port of aggregator", type=int, default="8080",
    )
    parser.add_argument(
        "-f", "--upload-folder", help="Path to upload files", type=str, required=True
    )
    parser.add_argument(
        "--disable-he",
        help="Disable to use homomorphic encryption",
        action="store_true",
    )
    parser.add_argument("-k", "--client", help="Number of worker", type=int, default=2)
    parser.add_argument("-r", "--round", help="Number of round", type=int, default=3)

    print(parser.parse_args())
    return parser.parse_args()


# App config
def init_app(args):
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = path.join(
        app.root_path, f"uploads/{args.upload_folder}"
    )

    try:
        mkdir(app.config["UPLOAD_FOLDER"])
    except OSError as e:
        print("Folder already")

    app.config["GLOBAL_MODEL_NAME"] = "global_model.pickle"
    app.config["AGGREGATED_MODEL_NAME"] = "aggregated_model.pickle"
    app.config["GLOBAL_MODEL_NAME_TMP"] = "global_model.pickle.tmp"
    app.config["AGGREGATED_MODEL_NAME_TMP"] = "aggregated_model.pickle.tmp"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{args.upload_folder}-{args.client}-{args.round}.sqlite"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
    app.config["N_ROUND"] = args.round
    app.config["N_CLIENT"] = args.client
    app.config["USE_HE"] = not args.disable_he
    app.config["PORT"] = args.port
    return app


def clear_weights(keep_aggregated=False, round=""):
    global app
    files = glob(path.join(app.config["UPLOAD_FOLDER"], "*"))
    for f in files:
        if (
            keep_aggregated == True
            and f"{round}_" + app.config["AGGREGATED_MODEL_NAME"] in f
        ):
            continue
        else:
            remove(f)


def read_weights(filepath: str) -> Tuple[Dict, int, str]:
    use_he = app.config["USE_HE"]
    with open(filepath, "rb") as file:
        with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            # Get number of sample
            regex = r"(?P<round>\d)_\d_(?P<n_sample>\d*).weight.pickle"
            match = re.match(regex, path.split(filepath)[-1])
            number_sample = match.groupdict()["n_sample"]
            round_id = match.groupdict()["round"]

            # Read weights
            weights = mmap_obj.read()
            weights = Utils.deserialize(weights)
            if use_he:
                context = Utils.load_key()
                weights = Utils.decryptialize(context, weights)
        return weights, int(number_sample), round_id


def read_sum_sample(filenames: List[str]) -> int:
    sum_sample = 0
    for filename in filenames:
        regex = r"(?P<round>\d)_\d_(?P<n_sample>\d*).weight.pickle"
        number_sample = re.match(regex, path.split(filename)[-1]).groupdict()[
            "n_sample"
        ]
        sum_sample += int(number_sample)
    return sum_sample


args = init_args()
app = init_app(args)
clear_weights()


def FedAvg(round):
    global_weights = None
    files = glob(path.join(app.config["UPLOAD_FOLDER"], "*_*.weight.pickle"))
    sum_sample = read_sum_sample(files)
    round_id = round
    for f in files:
        if global_weights == None:
            global_weights, n_sample, client_round_id = read_weights(f)
            # global_weights = deepcopy(client_weights)
            for key, value in global_weights.items():
                global_weights[key] = value * (n_sample / sum_sample)
        else:
            client_weights, n_sample, client_round_id = read_weights(f)
            for key, value in client_weights.items():
                global_weights[key] += value * (n_sample / sum_sample)
            del client_weights
            gc.collect()
        if client_round_id != round_id:
            print("Round confusion")
            raise ValueError
        round_id = client_round_id
    if app.config["USE_HE"]:
        context = Utils.load_key()
        global_weights = Utils.encryptialize(context, global_weights, "fedavg")
        gc.collect()

    global_weights = Utils.serialize(global_weights)
    gc.collect()
    upload_directory = app.config["UPLOAD_FOLDER"]
    upload_name = secure_filename(f"{round_id}_" + app.config["AGGREGATED_MODEL_NAME"])
    filepath_to_upload = path.join(upload_directory, upload_name)

    with open(filepath_to_upload + ".tmp", "wb") as file:
        file.write(global_weights)
    del global_weights
    gc.collect()

    while not path.exists(filepath_to_upload + ".tmp"):
        sleep(5)
    rename(filepath_to_upload + ".tmp", filepath_to_upload)
    clear_weights(keep_aggregated=True, round=round_id)


# Init database
db.init_app(app)
with app.app_context():
    # Create tables
    db.drop_all()
    db.create_all()

    # Add new train profiles
    db.session.add(TrainProfile(app.config["N_CLIENT"], app.config["N_ROUND"]))
    db.session.commit()


def create_success_json_response(message: str, data):
    return make_response(
        jsonify({"success": True, "message": message, "data": data}), 200,
    )


def create_fail_json_response(reason: str):
    return make_response(jsonify({"success": True, "message": reason}), 500,)


@app.post("/register")
def register():
    with app.app_context():
        # Get last train profile
        profile = TrainProfile.query.order_by(TrainProfile.id.desc()).first()

        # Return fail if number of client is maximun
        # if profile.index == profile.n_client - 1:
        #     return create_fail_json_response(reason="Worker is enough")
        # # Else increase index and return client id
        # else:
        profile.index += 1
        db.session.commit()

        data_to_send = {
            "client_id": profile.index,
            "n_round": profile.n_round,
            "n_client": profile.n_client,
            "is_leader": profile.index == 0,
        }

        return create_success_json_response(
            message="Register success", data=data_to_send,
        )


@app.route("/model/global", methods=["POST", "GET"])
def model_global():
    upload_directory = app.config["UPLOAD_FOLDER"]
    upload_name = secure_filename(app.config["GLOBAL_MODEL_NAME"])
    filepath_to_upload = path.join(upload_directory, upload_name)

    if request.method == "POST":
        f = request.files["model"]
        # with open(filepath_to_upload + ".tmp", "wb") as file:
        #     chunk_size = 1024
        #     while True:
        #         chunk = request.stream.read(chunk_size)
        #         if len(chunk) == 0:
        #             file.close()
        #             break
        #         file.write(chunk)
        f.save(filepath_to_upload + ".tmp")
        f.close()

        del f
        gc.collect()

        while not path.exists(filepath_to_upload + ".tmp"):
            sleep(5)
        rename(filepath_to_upload + ".tmp", filepath_to_upload)

        message = "Upload first model successfully"
        return create_success_json_response(message=message, data=False)

    elif request.method == "GET":
        while not path.exists(filepath_to_upload):
            sleep(5)
        return send_from_directory(directory=upload_directory, path=upload_name)


@app.route("/model/aggregation/<round>", methods=["POST", "GET"])
def model_aggregation(round):
    if request.method == "POST":
        f = request.files["model"]

        upload_directory = app.config["UPLOAD_FOLDER"]
        upload_name = secure_filename(f.filename)
        filepath_to_upload = path.join(upload_directory, upload_name)

        # with open(filepath_to_upload + ".tmp", "wb") as file:
        #     chunk_size = 4096
        #     while True:
        #         chunk = request.stream.read(chunk_size)
        #         if len(chunk) == 0:
        #             file.close()
        #             break
        #         file.write(chunk)

        f.save(filepath_to_upload + ".tmp")
        f.close()

        del f
        gc.collect()

        while not path.exists(filepath_to_upload + ".tmp"):
            sleep(5)
        rename(filepath_to_upload + ".tmp", filepath_to_upload)

        while not path.exists(filepath_to_upload):
            sleep(5)

        # Do check until enough
        files = glob(path.join(upload_directory, f"{round}_*_*.weight.pickle"))
        if len(files) == app.config["N_CLIENT"]:
            FedAvg(round)
        gc.collect()

        message = "Upload first model successfully"
        return create_success_json_response(message=message, data=False)

    elif request.method == "GET":
        upload_directory = app.config["UPLOAD_FOLDER"]
        upload_name = secure_filename(f"{round}_" + app.config["AGGREGATED_MODEL_NAME"])
        filepath_to_upload = path.join(upload_directory, upload_name)

        # # Do check until enough
        # files = glob(path.join(upload_directory, "*_*.weight.pickle"))
        # while len(files) < app.config["N_CLIENT"]:
        #     files = glob(path.join(upload_directory, "*_*.weight.pickle"))

        while not path.exists(filepath_to_upload):
            sleep(5)
        return send_from_directory(directory=upload_directory, path=upload_name)
        # else:
        #     return create_fail_json_response("File not extsits")


if __name__ == "__main__":
    app.run(port=args.port)
