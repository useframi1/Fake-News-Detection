import pandas as pd
from fake_news_classifier import FakeNewsClassifier
from utils.utils import load_config


class ModelTrainingPipeline:
    def __init__(
        self,
        config_path: str,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
    ):
        self.config = load_config(config_path)
        self.train_data = train_data
        self.val_data = val_data

    def run(self):
        print(f"{"#" * 20} Model Training Pipeline  {"#" * 20}")

        print(f"{"="*5} Initializing model... {"="*5}")
        model = FakeNewsClassifier(
            self.config["classifier_config_path"], self.train_data["label"].nunique()
        )

        print(f"Using Device: {model.device}")
        print(f"Model Name: {model.config['model_name']}")

        print(f"{"="*5} Training model... {"="*5}")
        model.fit(
            self.train_data["content"],
            self.train_data["label"],
            self.val_data["content"],
            self.val_data["label"],
        )

        print(f"{"="*5} Saving model... {"="*5}")
        model.save()

        print("Model training pipeline completed.")
        print("#" * 50)
        print("\n")

        return model
