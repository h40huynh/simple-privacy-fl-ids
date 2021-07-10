from typing import Dict
from tenseal.tensors.ckkstensor import CKKSTensor
import torch
import pickle
import torch.nn as nn

from copy import deepcopy
from io import BytesIO
from requests import Session, Request, Response
from tenseal import Context, ckks_tensor, ckks_tensor_from, context_from


class Utils:
    @staticmethod
    def load_key() -> Context:
        context = None
        with open("../keys/seal.pub", "rb") as handle:
            context = context_from(handle.read())
        return context

    @staticmethod
    def encryptialize(
        context: Context, weight: Dict[str, CKKSTensor], strategy: str, is_batch=True,
    ) -> Dict[str, bytes]:
        encrypted_weight = dict()
        for key, value in weight.items():
            if strategy.lower() == "fedbn" and "bn" in key:
                continue
            encrypted_weight[key] = value.serialize()
        return encrypted_weight

    @staticmethod
    def decryptialize(
        context: Context, weight: Dict[str, bytes]
    ) -> Dict[str, CKKSTensor]:
        for key, value in weight.items():
            weight[key] = ckks_tensor_from(context, value)
        return weight

    @staticmethod
    def serialize(weight: Dict) -> bytes:
        return pickle.dumps(weight)

    @staticmethod
    def deserialize(weight: bytes) -> Dict:
        return pickle.loads(weight)
