import torch
import pickle
import torch.nn as nn

from typing import Any, Dict
from io import BytesIO
from tenseal import Context, ckks_tensor, ckks_tensor_from, context_from
from tqdm import tqdm


class Utils:
    @staticmethod
    def load_key() -> Context:
        context = None
        with open("../keys/seal.cxt", "rb") as handle:
            context = context_from(handle.read())
        return context

    @staticmethod
    def encryptialize(
        context: Context, weight: Dict[str, torch.Tensor], strategy: str, is_batch=True,
    ) -> Dict[str, bytes]:
        encrypted_weight = dict()
        for key, value in tqdm(weight.items(), desc="Encrypting"):
            if strategy.lower() == "fedbn" and "bn" in key:
                continue
            encrypted_weight[key] = ckks_tensor(
                context, value, batch=is_batch
            ).serialize()
        return encrypted_weight

    @staticmethod
    def decryptialize(
        context: Context, weight: Dict[str, bytes]
    ) -> Dict[str, torch.Tensor]:
        for key, value in tqdm(weight.items(), desc="Decrypting"):
            weight[key] = ckks_tensor_from(context, value).decrypt().tolist()
            weight[key] = torch.Tensor(weight[key])
        return weight

    @staticmethod
    def serialize(weight: Dict, get_len=False) -> BytesIO:
        data = pickle.dumps(weight)
        if not get_len:
            return BytesIO(data)
        return BytesIO(data), len(data)

    @staticmethod
    def deserialize(weight: bytes) -> Dict[str, Any]:
        return pickle.loads(weight)
