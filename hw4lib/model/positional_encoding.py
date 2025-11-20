import torch
from torch import nn
import math

'''
TODO: Implement this Module.

Specification:
- Module should add positional information to input embeddings
- Uses sinusoidal position encodings as described in "Attention Is All You Need"
- Positional encoding matrix should have shape (1, max_len, d_model)
- Even indices use sine functions, odd indices use cosine functions
- Wavelengths form geometric progression from 2π to 10000·2π
- Encoding values should be on same device as input tensor
- Should handle any sequence length up to max_len
- Should raise error if input sequence length exceeds max_len
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        
        Steps:
        1. Call parent class constructor using super().__init__()
        2. Call create_pe_table to initialize positional encoding matrix
        """     
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.
        
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        
        Side Effects:
            - Initializes the positional encoding buffer 'pe' 
              of shape (1, max_len, d_model) (in order to broadcast with input tensor)
        """
        # TODO: Implement create_pe_table
        #raise NotImplementedError # Remove once implemented
        # pe = NotImplementedError
        # # Register as buffer to save with model state
        # self.register_buffer('pe', pe)

        # Create position indices (max_len, 1), column vector
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # Compute the frequency terms for even dimensions: shape (d_model/2,)
        # This encodes different wavelengths (should be unique for each)
        # Precompute the division term to make it a lot less messy, conert to a vectorized formula. 
        div_term = torch.exp( # we convert 1/10,000^(2i/d) to the exponential form of e^-ln(10000)
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # Initialize positional encoding table (max_len, d_model)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)

        # Apply sin to even indices (0, 2, 4, ...) and cos to odd indices (1, 3, 5, ...)
        pe[:, 0::2] = torch.sin(position * div_term) # get all even terms
        pe[:, 1::2] = torch.cos(position * div_term) # get all odd terms.

        # Add batch dimension so shape becomes (1, max_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)
        
        # Register as buffer to save with model state
        self.register_buffer('pe', pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length 
        """
        # TODO: Implement forward
        # Step 1: Get sequence length from input tensor
        seq_len = x.size(1)
        # Step 2: Verify sequence length doesn't exceed maximum length, raise error if it does
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}")
        # Step 3: Add positional encodings to input
        pe_slice = self.pe[:, :seq_len, :].to(x.device)
        return x + pe_slice
