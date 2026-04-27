import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=16, alpha=32, dropout=0.05):
        super().__init__()

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_A = nn.Linear(
            in_features,
            r,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.lora_B = nn.Linear(
            r,
            out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_output + lora_output


def inject_lora(model, target_modules=("q_proj", "v_proj"), r=16, alpha=32, dropout=0.05):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_modules:
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        else:
            inject_lora(module, target_modules, r=r, alpha=alpha, dropout=dropout)


def save_lora_weights(model, path):
    lora_state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    torch.save(lora_state, path)


def load_lora_weights(model, path):
    lora_state = torch.load(path, map_location="cpu")
    model.load_state_dict(lora_state, strict=False)