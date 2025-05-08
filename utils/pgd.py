import torch


class PGD:
    def __init__(self, model, epsilon=1.0, alpha=0.3, steps=3):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.emb_name = self._auto_detect_emb_name()
        self.emb_backup = {}
        self.grad_backup = {}

    def _auto_detect_emb_name(self):
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

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                # PGD step
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    # Projection step
                    param.data = self.project(name, param.data)

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.emb_backup:
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
