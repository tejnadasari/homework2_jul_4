from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding.
        """
        # Tokenize the input image - keep device consistent
        with torch.no_grad():
            # Get the device of the tokenizer
            device = next(self.tokenizer.parameters()).device
            x_device = x.to(device)
            
            tokens = self.tokenizer.encode_index(x_device.unsqueeze(0))  # Add batch dimension
            tokens = tokens.squeeze(0)  # Remove batch dimension: (h, w)
            h, w = tokens.shape
            
            # Simple compression: store dimensions + tokens
            compressed_data = []
            
            # Store dimensions (2 bytes each)
            compressed_data.extend([h >> 8, h & 0xFF, w >> 8, w & 0xFF])
            
            # Store flattened tokens (2 bytes each since tokens can be up to 1024)
            flat_tokens = tokens.flatten().cpu().numpy()
            for token in flat_tokens:
                token = int(token)
                compressed_data.extend([token >> 8, token & 0xFF])
            
            return bytes(compressed_data)

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        # Convert bytes back to data
        data = list(x)
        
        # Extract dimensions
        h = (data[0] << 8) | data[1]
        w = (data[2] << 8) | data[3]
        
        # Extract tokens
        token_data = data[4:]
        tokens = []
        for i in range(0, len(token_data), 2):
            if i + 1 < len(token_data):
                token = (token_data[i] << 8) | token_data[i + 1]
                tokens.append(token)
        
        # Reshape to (h, w) and use same device as tokenizer
        device = next(self.tokenizer.parameters()).device
        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).view(h, w)
        
        # Decode using tokenizer
        with torch.no_grad():
            # Add batch dimension for decoding
            tokens_batch = tokens_tensor.unsqueeze(0)  # (1, h, w)
            reconstructed = self.tokenizer.decode_index(tokens_batch)  # (1, H, W, 3)
            result = reconstructed.squeeze(0)  # Remove batch dimension: (H, W, 3)
            
            return result


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
