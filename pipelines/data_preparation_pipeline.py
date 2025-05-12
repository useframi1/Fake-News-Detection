import pandas as pd
from sklearn.model_selection import train_test_split
from utils.utils import load_config

# import openattack as oa
from tqdm import tqdm
from transformers import BertTokenizer


class DataPreparationPipeline:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.df = pd.DataFrame()

    def __load_data(self):
        self.df = pd.read_csv(self.config["data_path"])
        self.df = self.df[[self.config["content_column"], self.config["label_column"]]]

    def __apply_label_mapping(self):
        label_mapping = self.config["label_mapping"]
        self.df[self.config["label_column"]] = self.df[self.config["label_column"]].map(
            label_mapping
        )

    def __sample_data(self, datasets_dict):
        for key, value in datasets_dict.items():
            df = pd.concat([value["content"], value["label"]], axis=1)

            sampled_df = df.groupby(by=self.config["label_column"]).apply(
                lambda x: x.sample(
                    n=int(self.config["sample_size"]),
                    random_state=self.config["random_state"],
                ),
                include_groups=False,
            )

            sampled_df = sampled_df.reset_index(level=0)[self.df.columns]

            datasets_dict[key]["content"] = sampled_df[self.config["content_column"]]
            datasets_dict[key]["label"] = sampled_df[self.config["label_column"]]

    def __split_data(self):
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            self.df[self.config["content_column"]],
            self.df[self.config["label_column"]],
            test_size=self.config["test_size"],
            random_state=self.config["random_state"],
        )

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts,
            train_labels,
            test_size=self.config["val_size"],
            random_state=self.config["random_state"],
        )

        datasets_dict = {
            "train_data": {
                "content": train_texts,
                "label": train_labels,
            },
            "val_data": {
                "content": val_texts,
                "label": val_labels,
            },
            "test_data": {
                "content": test_texts,
                "label": test_labels,
            },
        }

        return datasets_dict

    def run(self) -> tuple[dict, dict, dict]:
        print(f"{"#" * 20} Data Preparation Pipeline  {"#" * 20}")

        print(f"{"="*5} Loading data... {"="*5}")
        self.__load_data()

        print(f"{"="*5} Applying label mapping... {"="*5}")
        self.__apply_label_mapping()

        print(f"{"="*5} Splitting data... {"="*5}")
        datasets_dict = self.__split_data()

        if self.config["do_sampling"]:
            print(f"{"="*5} Sampling data... {"="*5}")
            self.__sample_data(datasets_dict)

        print("Data preparation pipeline completed.")
        print("#" * 50)
        print("\n")

        return (
            datasets_dict["train_data"],
            datasets_dict["test_data"],
            datasets_dict["val_data"],
        )
