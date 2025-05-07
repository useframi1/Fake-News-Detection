import torch


class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}
        self.emb_name = self._auto_detect_emb_name()

    def _auto_detect_emb_name(self):
        # Try to auto-detect the embedding weight name (e.g., 'bert.embeddings.word_embeddings.weight')
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and "embeddings.word_embeddings" in name
                and "weight" in name
            ):
                return name
        raise ValueError(
            "No embedding layer found with 'embeddings.word_embeddings' in name"
        )

    def attack(self):
        for name, param in self.model.named_parameters():
            if name == self.emb_name:
                # if param.grad is None:
                #     continue  # Skip if gradient is None
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
