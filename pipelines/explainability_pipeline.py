from utils.utils import load_config, set_device
import shap
import numpy as np
from transformers import BertTokenizer
from collections import defaultdict


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
        self.num_explain_samples = self.config["num_explain_samples"]

        self.contents = val_data["content"]
        self.labels = val_data["label"]
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def run_shap(self):
        print("Starting SHAP explainability analysis...")
        label_map = {
            0: "reliable",
            1: "bias",
            2: "conspiracy",
            3: "fake",
            4: "rumor",
            5: "unreliable",
            6: "other",
        }
        class_contents = {i: [] for i in range(len(label_map))}
        for content, label in zip(self.contents, self.labels):
            class_contents[label].append(content)
        samples_per_class = max(1, self.num_explain_samples // len(label_map))
        sampled_contents = []
        for class_label, class_data in class_contents.items():
            if class_data:
                num_samples = min(samples_per_class, len(class_data))
                sampled_contents.extend(
                    np.random.choice(class_data, size=num_samples, replace=False)
                )
        masker = shap.maskers.Text(self.tokenizer)
        output_names = [label_map[i] for i in range(len(label_map))]
        explainer = shap.Explainer(
            self.model.predict_proba, masker, output_names=output_names
        )
        shap_values = explainer(sampled_contents)
        print("SHAP values computed successfully.")
        return shap_values

    def get_top_contributing_words(self, shap_values):
        token_contributions = defaultdict(
            lambda: defaultdict(float)
        )  # token_contributions[token][class_label]

        # Token to ignore
        IGNORE_TOKEN = " [SEP]"

        for sv in shap_values:
            tokens = sv.data
            values = sv.values  # shape: (num_tokens, num_classes)
            for token_idx, token in enumerate(tokens):
                # Skip the [SEP] token
                if token == IGNORE_TOKEN:
                    continue
                for class_idx, value in enumerate(values[token_idx]):
                    # Only add positive contributions
                    if value > 0:
                        token_contributions[token][
                            class_idx
                        ] += value  # Only accumulate positive values

        # === Prepare label names ===
        label_map = {
            0: "reliable",
            1: "bias",
            2: "conspiracy",
            3: "fake",
            4: "rumor",
            5: "unreliable",
            6: "other",
        }

        # === Print and create word clouds for top positive contributing tokens ===
        for class_idx in range(len(label_map)):
            # Extract word importance scores
            token_scores = {
                token: contribs[class_idx]
                for token, contribs in token_contributions.items()
            }

            # Remove [SEP] token if it somehow made it into the scores
            if IGNORE_TOKEN in token_scores:
                del token_scores[IGNORE_TOKEN]

            # Sort tokens by contribution value
            sorted_items = sorted(
                token_scores.items(), key=lambda x: x[1], reverse=True
            )

            # Print the top 10 positive contributing tokens for this class
            print(
                f"\n--- Top 10 tokens positively contributing to label: {label_map[class_idx]} ---"
            )
            for i, (token, score) in enumerate(sorted_items[:10], 1):
                print(f"{i}. {token}: {score:.6f}")

    def run(self):
        shap_values = self.run_shap()
        self.get_top_contributing_words(shap_values)
