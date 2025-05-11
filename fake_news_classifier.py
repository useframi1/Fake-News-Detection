import pandas as pd
import plotly.graph_objects as go
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertTokenizer
from data.fake_news_dataset import FakeNewsDataset
from data.testing_dataset import TestingDataset
from torch.utils.data import DataLoader
from utils.utils import (
    load_config,
    get_model_architecture,
    get_criterion,
    get_optimizer,
    set_device,
    load_model,
    get_tokenizer,
    get_adversarial_training_method,
)
from utils.fgm import FGM
from utils.pgd import PGD
from datetime import datetime
from tqdm import tqdm


class FakeNewsClassifier:
    def __init__(self, config_path: str, num_classes: int = 10):
        try:
            self.config = load_config(config_path)
            self.num_classes = num_classes
            self.model_config = load_config(
                self.config["model_config_path"].format(
                    model_name=self.config["model_name"]
                )
            )
            self.device = set_device()
            self.classifier = get_model_architecture(
                self.config["model_name"], self.model_config, num_classes
            ).to(self.device)
            self.criterion = get_criterion(self.model_config["criterion"])
            self.learning_rate = self.model_config["learning_rate"]
            self.optimizer = get_optimizer(
                self.classifier,
                self.model_config["optimizer"],
                self.learning_rate,
                self.model_config["weight_decay"],
            )
            self.epochs = self.model_config["epochs"]
            self.tokenizer = get_tokenizer(
                self.config["model_name"], self.model_config["tokenizer"]
            )
            self.history = pd.DataFrame(columns=["epoch", "train_acc", "val_acc"])
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=self.model_config["lr_scheduler"]["mode"],
                patience=self.model_config["lr_scheduler"]["patience"],
                factor=self.model_config["lr_scheduler"]["factor"],
            )
            self.do_adversarial_training = self.model_config["do_adversarial_training"]
            if self.do_adversarial_training:
                self.adverarial_training_method = self.model_config[
                    "adversarial_training"
                ]["method"]
                self.adversarial_training_config = self.model_config[
                    "adversarial_training"
                ]["config"]
                self.adversarial_model = get_adversarial_training_method(
                    self.adverarial_training_method,
                    self.classifier,
                    **self.adversarial_training_config,
                )

            torch.cuda.empty_cache()
        except Exception as e:
            print(e)

    def __create_dataloaders(self, train_texts, train_labels, val_texts, val_labels):
        train_dataset = FakeNewsDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["loader_batch_size"],
            shuffle=True,
            pin_memory=False,
            num_workers=4,
        )

        if val_texts is None or val_labels is None:
            return train_loader, None

        val_dataset = FakeNewsDataset(val_texts, val_labels, self.tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["loader_batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=4,
        )

        return train_loader, val_loader

    def __train_phase(self, train_loader, device):
        self.classifier.train()
        total_loss, total_correct = 0, 0

        progress_bar = tqdm(
            train_loader, desc="Training", leave=True, dynamic_ncols=True
        )

        for batch in progress_bar:
            input_ids, attention_mask, token_type_ids, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
                batch["label"].to(device),
            )

            self.optimizer.zero_grad()
            outputs = self.classifier(input_ids, attention_mask, token_type_ids)

            loss = self.criterion(outputs, labels)
            loss.backward()

            if self.do_adversarial_training:
                self.adversarial_model.run(
                    self.criterion, input_ids, attention_mask, token_type_ids, labels
                )

            self.optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

            avg_loss = total_loss / (progress_bar.n + 1)
            avg_acc = total_correct / ((progress_bar.n + 1) * train_loader.batch_size)

            progress_bar.set_postfix(loss=avg_loss, acc=avg_acc)

        progress_bar.close()

        total_loss /= len(train_loader)
        train_acc = total_correct / len(train_loader.dataset)

        return train_acc

    def __validate_phase(self, val_loader, device):
        self.classifier.eval()
        val_loss, val_correct = 0, 0

        progress_bar = tqdm(
            val_loader, desc="Validating", leave=True, dynamic_ncols=True
        )

        with torch.no_grad():
            for batch in progress_bar:
                input_ids, attention_mask, token_type_ids, labels = (
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["token_type_ids"].to(device),
                    batch["label"].to(device),
                )

                outputs = self.classifier(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

                avg_loss = val_loss / (progress_bar.n + 1)
                avg_acc = val_correct / ((progress_bar.n + 1) * val_loader.batch_size)

                progress_bar.set_postfix(loss=avg_loss, acc=avg_acc)

        progress_bar.close()

        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)

        return val_acc

    def fit(
        self,
        train_texts: pd.Series,
        train_labels: pd.Series,
        val_texts: pd.Series = None,
        val_labels: pd.Series = None,
    ):
        train_loader, val_loader = self.__create_dataloaders(
            train_texts, train_labels, val_texts, val_labels
        )

        for epoch in range(self.epochs):
            start_time = datetime.now()
            print(f"Epoch {epoch + 1}/{self.epochs}")

            # Training phase
            train_acc = self.__train_phase(train_loader, self.device)

            torch.cuda.empty_cache()

            # Validation phase
            if val_loader is not None:
                val_acc = self.__validate_phase(val_loader, self.device)
                self.scheduler.step(val_acc)

                self.history.loc[len(self.history)] = {
                    "epoch": epoch + 1,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                }
            else:
                self.history.loc[len(self.history)] = {
                    "epoch": epoch + 1,
                    "train_acc": train_acc,
                }

            epoch_time = (datetime.now() - start_time).total_seconds() / 60

            print(
                f"\nTraining Accuracy: {train_acc:.4f}\t{f"Validation Accuracy: {val_acc:.4f}" if val_loader else ''}\n"
            )
            print(f"Epoch took {epoch_time:.2f} minutes to complete")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]}")

            print("=" * 25)

    def predict(self, texts):
        test_dataset = TestingDataset(texts, self.tokenizer)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["loader_batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=4,
        )

        predictions = []

        self.classifier.eval()
        torch.cuda.empty_cache()

        progress_bar = tqdm(test_loader, desc="Testing", leave=True, dynamic_ncols=True)

        with torch.no_grad():
            for batch in progress_bar:
                input_ids, attention_mask, token_type_ids = (
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    batch["token_type_ids"].to(self.device),
                )
                outputs = self.classifier(input_ids, attention_mask, token_type_ids)
                predictions.extend(outputs.argmax(dim=1).cpu().tolist())

            progress_bar.close()

        return predictions

    def predict_proba(self, texts):
        test_dataset = TestingDataset(texts, self.tokenizer)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["loader_batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

        probabilities = []

        self.classifier.eval()
        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, token_type_ids = (
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    batch["token_type_ids"].to(self.device),
                )
                outputs = self.classifier(input_ids, attention_mask, token_type_ids)
                probabilities.extend(
                    torch.nn.functional.softmax(outputs, dim=1).cpu().tolist()
                )

        return probabilities

    def save(self, model_path=None, metrics=None):
        model_data = {
            "model_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model_config,
            "metrics": metrics if metrics else {},
            "history": (
                self.history.to_dict(orient="records") if not self.history.empty else {}
            ),
        }
        path = self.config["model_path"].format(model_name=self.config["model_name"])
        if model_path is not None:
            path = model_path
        torch.save(model_data, path)
        print(f"Model saved to {path}")

    def plot_history(self):
        if self.history.empty:
            print("No accuracy data available. Train the model first.")
            return

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.history["epoch"],
                y=self.history["train_acc"],
                mode="lines+markers",
                name="Train Accuracy",
                line=dict(color="blue"),
            )
        )

        if "val_acc" in self.history.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.history["epoch"],
                    y=self.history["val_acc"],
                    mode="lines+markers",
                    name="Validation Accuracy",
                    line=dict(color="orange"),
                )
            )

        fig.update_layout(
            title="Accuracy progress Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            template="plotly_dark",
            legend=dict(x=0, y=1),
        )

        fig.show()

    def load_pretrained(self, model_path):
        try:
            model_dict = load_model(model_path)

            self.model_config = model_dict["config"]

            self.classifier = get_model_architecture(
                "bert-bilstm", self.model_config, self.num_classes
            ).to(self.device)

            self.classifier.load_state_dict(model_dict["model_state_dict"])  # weights
            self.optimizer.load_state_dict(model_dict["optimizer_state_dict"])
            self.metrics = model_dict.get("metrics", {})
            self.history = pd.DataFrame(model_dict.get("history", []))

            print(self.metrics)

            print(f"Model loaded successfully from {model_path}")
            return self  # Indicate successful loading

        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File {model_path} not found.")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
