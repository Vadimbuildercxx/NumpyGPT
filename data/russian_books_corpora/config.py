from dataclasses import dataclass
import cupy as cp

@dataclass
class GPTConfig:
    block_size: int = 512  # removed only in attention
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 10
    n_head: int = 12
    n_embd: int = 576
    dropout: float = 0.0  # 0.2
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    axis: int = -1
    dtype: cp.dtype = cp.float32