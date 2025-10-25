import abc
import torch


def load() -> torch.nn.Module:
    from pathlib import Path
    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    """(H, W, C) → (C, H, W)"""
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    """(C, H, W) → (H, W, C)"""
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


# ---------------------------------------------------------------------------

class PatchifyLinear(torch.nn.Module):
    """Patchify an image into embeddings"""
    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.patch_conv = torch.nn.Conv2d(3, latent_dim, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, 3)
        return chw_to_hwc(self.patch_conv(hwc_to_chw(x)))


class UnpatchifyLinear(torch.nn.Module):
    """Reconstruct image from patch embeddings"""
    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.unpatch_conv = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, h, w, latent_dim)
        return chw_to_hwc(self.unpatch_conv(hwc_to_chw(x)))


# ---------------------------------------------------------------------------

class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 3) → (B, h, w, bottleneck_dim)"""
    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, h, w, bottleneck_dim) → (B, H, W, 3)"""


# ---------------------------------------------------------------------------

class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """Simple convolutional patch-level autoencoder - NO activation functions as per TA advice"""
    def __init__(self, patch_size: int = 25, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.bottleneck = bottleneck
        
        self.patchify = PatchifyLinear(patch_size, latent_dim) # Need to call the forward function Assume this is the object 
        self.bottleneck_conv = torch.nn.Conv2d(latent_dim, bottleneck, 1)  
        self.expand_conv = torch.nn.Conv2d(bottleneck, latent_dim, 1)     
        self.unpatchify = UnpatchifyLinear(patch_size, latent_dim)


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchify.forward(x)  # (B, h, w, latent_dim)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unpatchify.forward(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        return reconstructed, {}
