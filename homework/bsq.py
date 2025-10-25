import abc
import torch
import torch.nn.functional as F
from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path
    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


# ---------------------------------------------------------------------------

def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """Differentiable sign with straight-through gradient."""
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


# ---------------------------------------------------------------------------

class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor: ...
    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------

class BSQ(torch.nn.Module):
    """Binary Spherical Quantization bottleneck"""
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self.embedding_dim = embedding_dim
        self.down_proj = torch.nn.Linear(embedding_dim, codebook_bits)
        self.up_proj = torch.nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_proj(x)
        x = x / torch.norm(x, dim=-1, keepdim=True)  # TA's improved L2 normalization
        x = diff_sign(x)
        # Remove second normalization to allow gradient flow
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # -------------------------------------------------------------------
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        codes = self.encode(x)
        return self._code_to_index(codes).long()

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        codes = self._index_to_code(x)
        return self.decode(codes)

    # -------------------------------------------------------------------
    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        bits = (x >= 0).int()
        powers = (2 ** torch.arange(x.size(-1), device=x.device)).int()
        return (bits * powers).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        # arithmetic (no bitwise &) for Apple MPS safety
        powers = (2 ** torch.arange(self._codebook_bits, device=x.device)).float()
        bits = ((x.unsqueeze(-1) // powers) % 2).float()
        return 2 * bits - 1


# ---------------------------------------------------------------------------

class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """PatchAutoEncoder + BSQ quantization/tokenizer"""
    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim, bottleneck=latent_dim)
        self.codebook_bits = codebook_bits
        self.bsq = BSQ(codebook_bits, latent_dim)

    # -------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = super().encode(x)
        return self.bsq.encode(encoded)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        decoded_features = self.bsq.decode(x)
        return super().decode(decoded_features)

    # -------------------------------------------------------------------
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        encoded = super().encode(x)
        return self.bsq.encode_index(encoded).long()

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        quantized = self.bsq.decode_index(x)
        return super().decode(quantized)

    # -------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)

        # compute codebook stats once, without re-encoding
        indices = self.bsq._code_to_index(encoded).flatten()
        cnt = torch.bincount(indices, minlength=2 ** self.codebook_bits)

        extras = {
            "cb0": (cnt == 0).float().mean().detach(),
            "cb1": (cnt == 1).float().mean().detach(),
            "cb2": (cnt <= 2).float().mean().detach(),
        }
        return reconstructed, extras
