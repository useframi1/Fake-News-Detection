import pandas as pd
from fake_news_classifier import FakeNewsClassifier
from utils.utils import load_config, set_device
import shap
import lime
import lime.lime_text
import numpy as np
import torch
from transformers import BertTokenizer
from data.testing_dataset import TestingDataset
from torch.utils.data import DataLoader


class ExplainabilityPipeline:
    def __init__(
        self,
        config_path: str,
        model,
        val_data: dict,
        tokenizer_name="bert-base-uncased",
    ):

        self.config = load_config(config_path)
        self.device = set_device()
        self.model = model
        self.val_data = val_data
        self.num_background = self.config["num_background"]
        self.num_explain_samples = self.config["num_explain_samples"]

        self.contents = val_data["content"]
        self.labels = val_data["label"]
        # Reset the index of the DataFrame or Series
        self.contents = self.contents.reset_index(drop=True)
        self.labels = self.labels.reset_index(drop=True)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def run_shap(self):

        print("Starting SHAP explainability analysis...")

        idx = np.random.choice(len(self.contents), self.num_background, replace=False)
        background_contents = [self.contents[i] for i in idx]

        # Create a masker for text
        masker = shap.maskers.Text(self.tokenizer)

        # Initialize SHAP Explainer
        explainer = shap.Explainer(self.predict_proba, masker)

        # Explain the first few contents
        explain_contents = background_contents[: self.num_explain_samples]
        shap_values = explainer(explain_contents)

        print("SHAP values computed successfully.")
        for i in range(min(len(explain_contents), 3)):  # Show first 3 examples
            self.visualize_single_shap(shap_values[i], i)

    def run_lime(self):

        print("Starting LIME explainability analysis...")

        explain_contents = self.contents[: self.num_explain_samples]
        explain_inputs = self.tokenize_inputs(explain_contents)
        lime_explainer = lime.lime_text.LimeTextExplainer(
            class_names=[str(i) for i in range(7)]
        )

        for i, content in enumerate(explain_contents):
            print(f"Explaining sample {i+1}/{len(explain_contents)}...")
            explanation = lime_explainer.explain_instance(
                content, self.predict_lime, num_features=10
            )

            print(explanation.as_list())

    def predict_proba(self, texts):
        test_dataset = TestingDataset(texts, self.tokenizer)
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

        probabilities = []

        self.model.eval()

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, token_type_ids = (
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    batch["token_type_ids"].to(self.device),
                )
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                probabilities.extend(
                    torch.nn.functional.softmax(outputs, dim=1).cpu().tolist()
                )

        return probabilities

    def tokenize_inputs(self, contents):

        text_inputs = self.tokenizer(
            contents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        # Return a list of tokenized inputs
        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
        }

    def predict_lime(self, texts):

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
            add_special_tokens=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        return probs.numpy()

    def visualize_single_shap(self, shap_values, index):
        print(f"Visualizing SHAP values for sample {index+1}...")

        # html = shap.plots.text(shap_values)
        # with open(f"shap_sample_{index+1}.html", "w") as f:
        #     f.write(html)
        # print(f"SHAP visualization saved as 'shap_sample_{index+1}.html'")

        # Print top contributing tokens
        token_importance = list(zip(shap_values.data, shap_values.values))
        token_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"Top 10 important tokens for sample {index+1}:")
        for token, importance in token_importance[:10]:
            print(f"Token: '{token}', Importance: {importance:.4f}")
