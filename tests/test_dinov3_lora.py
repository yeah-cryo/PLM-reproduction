import torch
from torch import nn

from plm.model import LoRALinear, apply_attention_lora


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([Layer(), Layer()])


def test_lora_starts_as_exact_base_projection():
    base = nn.Linear(4, 3)
    adapter = LoRALinear(base, rank=2, alpha=4)
    inputs = torch.randn(5, 4)
    torch.testing.assert_close(adapter(inputs), base(inputs))
    assert adapter.weight.requires_grad is False
    assert adapter.lora_A.requires_grad and adapter.lora_B.requires_grad


def test_attention_lora_respects_layer_range():
    model = Encoder()
    names = apply_attention_lora(model, start_layer=2, end_layer=2)
    assert len(names) == 4
    assert isinstance(model.layer[0].attention.q_proj, nn.Linear)
    assert isinstance(model.layer[1].attention.q_proj, LoRALinear)
