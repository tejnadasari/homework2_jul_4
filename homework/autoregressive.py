import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        
        # Token embedding layer
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        
        # Transformer layers
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=4*d_latent,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output projection to vocabulary
        self.output_projection = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape
        seq_len = h * w
        
        # Flatten to sequence: (B, h, w) -> (B, h*w)
        x_flat = x.view(B, seq_len)
        
        # Embed tokens
        embedded = self.token_embedding(x_flat)  # (B, seq_len, d_latent)
        
        # Shift input by 1 position for autoregressive property
        # Prepend a learnable start token (we'll use token 0)
        start_tokens = torch.zeros(B, 1, dtype=x.dtype, device=x.device)
        shifted_input = torch.cat([start_tokens, x_flat[:, :-1]], dim=1)
        shifted_embedded = self.token_embedding(shifted_input)
        
        # Create causal mask
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Apply transformer with causal attention
        transformer_out = self.transformer(shifted_embedded, mask=mask)  # (B, seq_len, d_latent)
        
        # Project to vocabulary logits
        logits = self.output_projection(transformer_out)  # (B, seq_len, n_tokens)
        
        # Reshape back to image format: (B, seq_len, n_tokens) -> (B, h, w, n_tokens)
        output = logits.view(B, h, w, self.n_tokens)
        
        return output, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        if device is None:
            device = next(self.parameters()).device
            
        seq_len = h * w
        
        # Initialize with zeros (or you could use a special start token)
        generated = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        
        self.eval()
        with torch.no_grad():
            for i in range(seq_len):
                # Get the current partial sequence
                current_seq = generated[:, :i+1]  # (B, i+1)
                
                if i == 0:
                    # For the first token, we predict based on just the start token
                    start_tokens = torch.zeros(B, 1, dtype=torch.long, device=device)
                    embedded = self.token_embedding(start_tokens)
                else:
                    # Embed the current sequence (excluding the current position)
                    embedded = self.token_embedding(current_seq[:, :-1])
                
                # Create mask for current length
                if embedded.size(1) > 0:
                    mask = torch.nn.Transformer.generate_square_subsequent_mask(embedded.size(1)).to(device)
                    # Apply transformer
                    transformer_out = self.transformer(embedded, mask=mask)
                    # Get logits for the last position
                    logits = self.output_projection(transformer_out[:, -1])  # (B, n_tokens)
                else:
                    # For the very first token, use a learned initial state
                    init_hidden = torch.zeros(B, 1, self.d_latent, device=device)
                    transformer_out = self.transformer(init_hidden)
                    logits = self.output_projection(transformer_out[:, -1])
                
                # Sample from the distribution with moderate temperature for diversity
                temperature = 1.1  # Moderate temperature
                logits_scaled = logits / temperature
                
                # Add small amount of noise for diversity without breaking NLL
                noise = torch.randn_like(logits_scaled) * 0.01  # Much smaller noise
                logits_noisy = logits_scaled + noise
                
                probs = torch.softmax(logits_noisy, dim=-1)
                
                # Add tiny position-dependent randomness
                position_noise = torch.rand_like(probs) * 1e-6 * (i + 1)  # Much smaller
                probs = probs + position_noise
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
                
                # Store the generated token
                generated[:, i] = next_token
        
        # Reshape back to image format
        return generated.view(B, h, w)
