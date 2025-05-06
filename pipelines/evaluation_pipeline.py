import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
)
from fake_news_classifier import FakeNewsClassifier
from utils.utils import load_config, load_model


class EvaluationPipeline:
    def __init__(
        self,
        config_path: str,
        test_data: pd.DataFrame,
        model: FakeNewsClassifier,
    ):
        self.config = load_config(config_path)
        self.test_texts = test_data["content"]
        self.test_labels = test_data["label"]
        self.model = model

    def __get_model_predictions(self):
        predictions = self.model.predict(self.test_texts)
        return predictions

    def __compute_metrics(self, predictions):
        accuracy = accuracy_score(self.test_labels, predictions)
        precision = precision_score(
            self.test_labels, predictions, average="weighted", zero_division=0
        )
        recall = recall_score(
            self.test_labels, predictions, average="weighted", zero_division=0
        )
        f1 = f1_score(
            self.test_labels, predictions, average="weighted", zero_division=0
        )

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        return metrics_dict

    def __display_metrics(self, metrics_dict, predictions):
        for metric, value in metrics_dict.items():
            print(f"{metric}: {value:.4f}")

        report = classification_report(self.test_labels, predictions, zero_division=0)

        print("\nClassification Report:\n", report)

    def __get_best_model_metrics(self):
        try:
            model = load_model(self.config["best_model_path"])
            return model["metrics"]
        except FileNotFoundError as e:
            print(e)
            return None

    def __compare_models(self, current_metrics, best_metrics):
        for metric in current_metrics:
            current_value = current_metrics[metric]
            best_value = best_metrics.get(metric, None)

            if best_value is None:
                print(f"{metric}: Current = {current_value:.4f}, Best = N/A")
            else:
                change = current_value - best_value
                status = "Improved" if change > 0 else "Worse" if change < 0 else "Same"
                print(
                    f"{metric}: Current = {current_value:.4f}, Best = {best_value:.4f} ({status})"
                )

        print("\nComparison Complete")

    def __compare_models(self, current_metrics, best_metrics):
        return current_metrics["accuracy"] > best_metrics["accuracy"]

    def __save_best_model(self, model_metrics):
        best_metrics = self.__get_best_model_metrics()
        if best_metrics is None or self.__compare_models(model_metrics, best_metrics):
            self.model.save(self.config["best_model_path"], model_metrics)
            print("\nNew best model saved.")
        else:
            print("\nCurrent model is not better than the best model.")

    def run(self):
        print(f"{"#" * 20} Evaluation Pipeline {"#" * 20}")

        predictions = self.__get_model_predictions()

        model_metrics = self.__compute_metrics(predictions)

        self.__display_metrics(model_metrics, predictions)

        self.model.plot_history()

        self.model.save(metrics=model_metrics)

        self.__save_best_model(model_metrics)
