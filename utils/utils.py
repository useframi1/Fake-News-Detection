import json
import torch
from transformers import BertTokenizer, RobertaTokenizer
from model_architectures.bert_bilstm_classifier import BertBiLSTMClassifier
from model_architectures.bert_classifier import BertClassifier
from model_architectures.roberta_bilstm_classifier import RobertaBiLSTMClassifier


def load_config(config_path) -> dict:
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file not found.")


def get_model_architecture(model_name, config, num_classes):
    if model_name == "bert-bilstm":
        return BertBiLSTMClassifier(config, num_classes)
    elif model_name == "roberta-bilstm":
        return RobertaBiLSTMClassifier(config, num_classes)
    elif model_name == "bert":
        return BertClassifier(config, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_tokenizer(model_name, tokenizer_name):
    if model_name == "bert-bilstm":
        return BertTokenizer.from_pretrained(tokenizer_name)
    elif model_name == "roberta-bilstm":
        return RobertaTokenizer.from_pretrained(tokenizer_name)
    elif model_name == "bert":
        return BertTokenizer.from_pretrained(tokenizer_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_criterion(criterion_name):
    if criterion_name == "cross-entropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")


def get_optimizer(model, optimizer_name, learning_rate, weight_decay):
    if optimizer_name == "adamW":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def load_model(model_path):
    try:
        checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")

        # config = checkpoint.get("config", {})
        # model = FakeNewsClassifier(
        #     config["classifier_config_path"], config["num_classes"]
        # )
        # model.classifier.load_state_dict(checkpoint["model_state_dict"])

        # optimizer = get_optimizer(
        #     model.classifier, config["optimizer"], config["learning_rate"]
        # )
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        model_dict = {
            "model_state_dict": checkpoint["model_state_dict"],
            "optimizer_state_dict": checkpoint["optimizer_state_dict"],
            "config": checkpoint["config"],
            "metrics": checkpoint.get("metrics", {}),
            "history": checkpoint.get("history", {}),
        }

        return model_dict
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File {model_path} not found.")


def get_adversarial_training_method(method_name, model, **kwargs):
    if method_name == "fgm":
        from utils.fgm import FGM

        return FGM(model, **kwargs)
    elif method_name == "pgd":
        from utils.pgd import PGD

        return PGD(model, **kwargs)
    else:
        raise ValueError(f"Unsupported adversarial training method: {method_name}")
