import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        
        batch_size = x.size(0) # intialize 
        scores = torch.zeros(batch_size, device=x.device) # Track accumulated log-prob scores
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)  # Track which sequences hit EOS

        max_steps = self.max_length - x.size(1)
        for _ in range(max_steps):
            if finished.all():
                break  # Stop early if every sequence is done

            logits = self.score_fn(x)  # (batch_size, vocab_size)
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)  # Penalize repeats
            logits = logits / temperature  # Apply temperature scaling
            log_probs = torch.log_softmax(logits, dim=-1)  # Convert to log probabilities

            next_tokens = torch.argmax(log_probs, dim=-1)  # Greedy pick per batch element
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)  # Scores for chosen tokens

            scores = torch.where(finished, scores, scores + token_scores)  # Update only unfinished sequences
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)  # Append next tokens to the sequences

            finished = finished | (next_tokens == self.tokenizer.eos_id)  # Mark sequences that generated EOS

        return x, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length) where each sequence in a beam set is sorted by score
             - scores is of shape (batch_size, beam_width)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        batch_size, seq_len = x.shape
        device = x.device

        # Initial logits for the first expansion
        logits = self.score_fn(x)  # (B, V)
        logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
        logits = logits / temperature
        log_probs = torch.log_softmax(logits, dim=-1)

        # Select top-k to seed beams
        top_log_probs, top_tokens = torch.topk(log_probs, k=beam_width, dim=-1)  # (B, K)

        # Initialize beams: expand x to (B, K, seq_len) and append first tokens -> (B, K, seq_len+1)
        beams = x.unsqueeze(1).expand(batch_size, beam_width, seq_len).clone()
        beams = torch.cat([beams, top_tokens.unsqueeze(-1)], dim=-1)

        # Scores and finished flags per beam
        scores = top_log_probs.clone()  # (B, K)
        finished = top_tokens.eq(self.tokenizer.eos_id)  # (B, K)

        vocab_size = log_probs.size(-1)
        max_steps = self.max_length - beams.size(-1)

        for _ in range(max_steps):
            # If all beams in all batches finished, stop early
            if finished.all():
                break

            # Collect next-token scores for each beam
            next_token_scores = []
            for b in range(beam_width):
                # Score current beam sequences; shape (B, V)
                logits_b = self.score_fn(beams[:, b, :])
                # Apply repeat penalty before temperature to mirror pseudocode order
                logits_b = self._apply_repeat_penalty(logits_b, beams[:, b, :], repeat_penalty)
                logits_b = logits_b / temperature
                next_token_scores.append(logits_b)

            # Stack to shape (B, K, V)
            next_token_scores = torch.stack(next_token_scores, dim=1)
            # Convert to log-probs
            next_log_probs = torch.log_softmax(next_token_scores, dim=-1)  # (B, K, V)

            # Prevent finished beams from changing: force them to stay on EOS with zero score delta
            eos_mask = finished.unsqueeze(-1)  # (B, K, 1)
            frozen = torch.full_like(next_log_probs, float('-inf'))
            frozen[..., self.tokenizer.eos_id] = 0.0
            next_log_probs = torch.where(eos_mask, frozen, next_log_probs)

            # Cumulative scores for all beam-token combinations
            cum_scores = scores.unsqueeze(-1) + next_log_probs  # (B, K, V)

            # Flatten beams to select top-k globally per batch: shape -> (B, K*V)
            flat_scores = cum_scores.view(batch_size, -1)
            topk_scores, topk_indices = torch.topk(flat_scores, k=beam_width, dim=-1)

            # Map flat indices back to beam and token
            beam_indices = topk_indices // vocab_size  # (B, K)
            next_tokens = topk_indices % vocab_size    # (B, K)

            # Reorder beams according to selected parents
            beams = beams.gather(
                dim=1,
                index=beam_indices.unsqueeze(-1).expand(-1, -1, beams.size(-1))
            )
            finished = finished.gather(dim=1, index=beam_indices)  # carry finished flags forward

            # Append chosen next tokens
            beams = torch.cat([beams, next_tokens.unsqueeze(-1)], dim=-1)  # (B, K, seq_len+1+step)

            # Update scores and finished flags
            scores = topk_scores
            finished = finished | next_tokens.eq(self.tokenizer.eos_id)

        # Sort beams within each batch by score descending
        sorted_scores, sort_idx = torch.sort(scores, dim=-1, descending=True)
        sorted_beams = beams.gather(
            dim=1,
            index=sort_idx.unsqueeze(-1).expand(-1, -1, beams.size(-1))
        )

        return sorted_beams, sorted_scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break

            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            # Check if any sequence has reached EOS 
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]
