"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass
import time

import numpy as np
from tqdm import tqdm

import cupy as cp


class LayerNorm:
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias, eps=1e-05,):
        super().__init__()
        self.weight = cp.ones(ndim)
        self.bias = cp.zeros(ndim) if bias else None
        self.weight_g = cp.zeros_like(self.weight)
        self.bias_g = cp.zeros_like(self.bias)

        self.eps = eps

        self.sqrt_var = None
        self.saved_normalized = None
        self.scaled_x = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        mean = x.mean(-1, keepdims=True) # (B, T, 1)
        var = x.var(-1, ddof=0, keepdims=True) # (B, T, 1)
        self.sqrt_var = cp.sqrt(var + self.eps) # (B, T, 1)
        normalized = (x - mean) / self.sqrt_var  #  (B, T, C)

        self.saved_normalized = normalized

        out = self.weight * self.saved_normalized + self.bias # (ndim) * (B, T, C) + (ndim) = (B, T, C)
        return out

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """
        Normalized matrix is already saved in saved_normalized and
        inverse of variance is already saved in saved_var_inv.
        :param grad: input gradient (B, T, C)
        :return: batch norm gradient
        """
        n = grad.shape[-1]

        self.weight_g += cp.sum(grad * self.saved_normalized, axis=(0, 1))
        self.bias_g += cp.sum(grad, axis=(0, 1))

        a = grad * self.weight
        b = cp.sum(a, axis=-1, keepdims=True) / n
        c = self.saved_normalized * cp.sum(self.saved_normalized * a, axis=-1, keepdims=True) / n

        grad_out = (a - b - c) / self.sqrt_var

        return grad_out

softmax_forward_kernel = cp.ElementwiseKernel(
    'T x, T m',
    'T t',
    """
        t = exp(x - m)
    """,
    'softmax_forward_kernel')

softmax_backward_sum_kernel = cp.ReductionKernel(
    'T g, T p',
    'T y',
    'g * p',
    'a = a + b',
    'y = a',
    identity='0',
    name='softmax_backward_sum_kernel'
)

softmax_backward_mm_kernel = cp.ElementwiseKernel(
    'T p, T g, T r',
    'T t',
    """
        t = p * (g - r)
    """,
    'softmax_backward_mm_kernel')

class Softmax:
    def __init__(self, axis):
        self.axis = axis
        self.probs = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        # exp_x = cp.exp(x - cp.expand_dims(cp.max(x, axis=self.axis), axis=self.axis))
        # self.probs = exp_x/cp.sum(exp_x, axis=self.axis, keepdims=True)
        max_x = cp.max(x, axis=self.axis, keepdims=True)
        exp_x = softmax_forward_kernel(x, max_x) #cp.exp(x - max_x)
        self.probs = exp_x/cp.sum(exp_x, axis=self.axis, keepdims=True)
        return self.probs

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        #out = (self.probs * (grad - (grad * self.probs).sum(axis=self.axis, keepdims=True, dtype=cp.float32))).astype(grad.dtype)
        return softmax_backward_mm_kernel(self.probs, grad, softmax_backward_sum_kernel(grad, self.probs, axis=-1, keepdims=True))


class Linear:
    def __init__(self, in_features, out_features, bias=True, dtype=cp.float32):
        self.W = cp.random.normal(0.0, 1.0, (out_features, in_features)).astype(dtype=dtype)
        self.b = cp.zeros(out_features, dtype=dtype)
        self.Wg = cp.zeros_like(self.W)
        self.bg = cp.zeros_like(self.b)
        self.saved_x = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.saved_x = x.copy()
        out = cp.matmul(self.saved_x, self.W.T) + self.b
        return out

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        dh = cp.matmul(grad, self.W)
        self.Wg += cp.matmul(cp.moveaxis(grad, -1, -2), self.saved_x).sum(0)
        self.bg += grad.sum((0, 1))
        return dh


class Dropout:
    def __init__(self, rate):
        assert 0 <= rate <= 1
        self.rate = rate
        self.saved_out = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.__saved_mask = (cp.random.rand(*x.shape) > self.rate)
        return self.__saved_mask * x / (1.0 - self.rate)

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        return self.__saved_mask / (1.0 - self.rate) * grad


class Embedding:
    def __init__(self, vocab_size, embedding_size, dtype=cp.float32):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weight = cp.random.normal(0.0, 1.0, (vocab_size, embedding_size)).astype(dtype=dtype)
        self.weight_g = cp.zeros_like(self.weight)
        self.saved_idx = None

    def forward(self, idx):
        self.saved_idx = idx.copy()
        out = self.weight[self.saved_idx] # replace to take
        return out

    def backward(self, grad):
        dweight = cp.zeros_like(self.weight)
        dweight[self.saved_idx] += cp.sum(grad, 0)
        self.weight_g += dweight
        return grad

gelu_tanh_forward_kernel = cp.ElementwiseKernel(
    'float64 x',
    'float64 t',
    """
        t = tanh(sqrt(2 / 3.141592653589793) * (x + 0.044715 * x*x*x))
    """,
    'gelu_tanh_forward_kernel')

gelu_backward_kernel = cp.ElementwiseKernel(
    'float64 x, float64 t',
    'float64 z',
    """
        z = 0.5 * (1 + t) + (1 - t * t) * x * (1 + 3 * 0.044715 * x * x) / sqrt(2 * 3.141592653589793)
    """,
    'gelu_backward_kernel')

class Gelu:
    r"""
    GELU (Gaussian Error Linear Units) module. Used formula:
    .. math::
       GELU(x) = 0.5 * x * (1 + Tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715 * x^3) ))
    """
    def __init__(self):
        self.c = 0.044715
        self.saved_in_x = None
        self.t = None

    def forward(self, x) -> cp.ndarray:
        self.saved_in_x = x.copy()
        self.t = gelu_tanh_forward_kernel(x) #cp.tanh(cp.sqrt(2 / cp.pi) * (x + self.c * x**3))
        out = 0.5 * x * (1 + self.t)
        return out

    def backward(self, grad) -> cp.ndarray:
        out = gelu_backward_kernel(self.saved_in_x, self.t) #0.5 * (1 + self.t) + (1 - self.t**2) * self.saved_in_x * (1 + 3 * self.c * self.saved_in_x ** 2) / math.sqrt(2 * math.pi)
        return out * grad


class MLP:

    def __init__(self, config):
        super().__init__()
        self.c_fc    = Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = Gelu()
        self.c_proj  = Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = Dropout(config.dropout)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        x = self.c_fc.forward(x)
        x = self.gelu.forward(x)
        x = self.c_proj.forward(x)
        x = self.dropout.forward(x)
        return x

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        grad = self.dropout.backward(grad)
        grad = self.c_proj.backward(grad)
        grad = self.gelu.backward(grad)
        grad = self.c_fc.backward(grad)
        return grad


class CausalSelfAttention:
    def __init__(self, config):
        # key, query, value projections for all heads, but in a batch
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)

        self.softmax = Softmax(axis=-1)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        #TODO: Check for block_size
        self.B, self.T, self.C = None, None, None
        self.att = None
        self.k_t = None
        self.q_t = None
        self.v_t = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward pass for CausalSelfAttention
        :param x:
        :return:
        """
        self.B, self.T, self.C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = cp.split(self.c_attn.forward(x), 3, axis=2)

        self.k_t = cp.transpose(k.reshape(self.B, self.T, self.n_head, self.C // self.n_head), (0, 2, 1, 3)) # (B, nh, T, hs)
        self.q_t = cp.transpose(q.reshape(self.B, self.T, self.n_head, self.C // self.n_head), (0, 2, 1, 3)) # (B, nh, T, hs)
        self.v_t = cp.transpose(v.reshape(self.B, self.T, self.n_head, self.C // self.n_head), (0, 2, 1, 3)) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        # manual implementation of attention
        att = cp.matmul(self.q_t,  cp.transpose(self.k_t, (0, 1, 3, 2))) * (1.0 / math.sqrt(self.k_t.shape[-1]))
        # causal mask to ensure that attention is only applied to the left in the input sequence
        att = cp.where(cp.tril(att, self.T - att.shape[-1]) != 0, att, float('-inf'))
        att = self.softmax.forward(att)
        self.att = self.attn_dropout.forward(att)

        y = cp.matmul(self.att, self.v_t) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = cp.ascontiguousarray(cp.transpose(y, (0, 2, 1, 3))).reshape(self.B, self.T, self.C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout.forward(self.c_proj.forward(y))
        return y

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """
        Backward pass for CausalSelfAttention
        :param grad:
        :return:
        """
        # output projection gradient
        c_proj_g = self.c_proj.backward(grad)
        resid_dropout_g = self.resid_dropout.backward(c_proj_g)
        y_g = resid_dropout_g.reshape(self.B, self.T, self.n_head, self.C // self.n_head)
        y_g = cp.transpose(y_g, (0, 2, 1, 3))

        v_g = cp.matmul(cp.transpose(self.att, (0, 1, 3, 2)), y_g) # QK^T @ y_g -> v_g: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        att_g = cp.matmul(y_g, cp.transpose(self.v_t, (0, 1, 3, 2))) # y_g @ v -> v_g: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        att_dropout_g = self.attn_dropout.backward(att_g)
        att_softmax_g = self.softmax.backward(att_dropout_g)
        att_softmax_g = cp.where(cp.tril(att_softmax_g, self.T - att_softmax_g.shape[-1]) != 0, att_softmax_g, 0)

        q_t_g = (1.0 / math.sqrt(self.k_t.shape[-1])) * (cp.matmul(att_softmax_g, self.k_t))
        k_t_g = (1.0 / math.sqrt(self.k_t.shape[-1])) * (cp.matmul(cp.transpose(att_softmax_g, (0, 1, 3, 2)), self.q_t)) # (Q^T @ att)^T = att^T @ Q

        k_t_g = cp.transpose(k_t_g, (0, 2, 1, 3)).reshape(self.B, self.T, self.C) # (B, nh, T, hs)
        q_t_g = cp.transpose(q_t_g, (0, 2, 1, 3)).reshape(self.B, self.T, self.C) # (B, nh, T, hs)
        v_g = cp.transpose(v_g, (0, 2, 1, 3)).reshape(self.B, self.T, self.C) # (B, nh, T, hs)

        split_g = cp.concatenate((q_t_g, k_t_g, v_g), axis=-1)
        x_g = self.c_attn.backward(split_g)
        return x_g


class Block:
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        x = x + self.attn.forward(self.ln_1.forward(x))
        x = x + self.mlp.forward(self.ln_2.forward(x))
        return x

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        grad_out = grad.copy()
        grad = self.mlp.backward(grad)
        grad = self.ln_2.backward(grad)
        grad_out += grad
        grad = self.attn.backward(grad_out)
        grad = self.ln_1.backward(grad)
        grad_out += grad
        return grad_out


class CrossEntropy:
    def __init__(self, axis=-1):
        self.axis = axis
        self.x = None
        self.y = None
        self.ignore_index = None
        self.idx = None

    def __log_softmax(self, x):
        c = x.max(axis=self.axis, keepdims=True)
        logsumexp = cp.log(cp.exp(x - c).sum(axis=self.axis, keepdims=True))
        return x - c - logsumexp

    def forward(self, x: cp.ndarray, y: cp.ndarray, ignore_index=-1):
        self.x = x
        self.y = y
        self.ignore_index = ignore_index
        #masked = cp.where(y != -1, cp.take_along_axis(self.__log_softmax(x), cp.expand_dims(y, 1), axis=1).T, 0)
        mask = (y != ignore_index)
        self.idx = (cp.arange(y.shape[0])[mask], y[mask])
        loss = (- 1. / cp.count_nonzero(y != -1)) * cp.sum(self.__log_softmax(x)[self.idx])
        return loss

    def backward(self, grad=None) -> cp.ndarray:
        max_x = cp.max(self.x, axis=self.axis, keepdims=True)
        exp_x = softmax_forward_kernel(self.x, max_x)
        probs = exp_x/cp.sum(exp_x, axis=self.axis, keepdims=True)
        probs[self.idx] -= 1.
        out = probs / cp.count_nonzero(self.y != self.ignore_index)
        out[self.y == self.ignore_index] = 0.
        return out


@dataclass
class GPTConfig:
    block_size: int = 256 # removed only in attention
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 768
    dropout: float = 0.2
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    axis: int = -1
    dtype: cp.dtype = cp.float32


class AdamW:
    """
    Implementation of Adam with weight decay from:
    https://arxiv.org/pdf/1711.05101v3
    https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    """
    def __init__(self, model, lr, betas: list = None, eps=1e-08,weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps

        number_of_tensors = model.get_number_of_tensors()
        self.mt: list = [.0 for _ in range(number_of_tensors)]
        self.vt: list = [.0 for _ in range(number_of_tensors)]
        self.t: list = [.0 for _ in range(number_of_tensors)]
        self.frozen_layers = []

    def freeze_layers(self, names):
        self.frozen_layers = names

    def step(self):
        model_params = self.model.named_parameter_optim_groups(self.weight_decay)
        idx = 0
        beta_1,  beta_2 = self.betas
        for param in model_params[0]["params"]:
            if param[2] not in self.frozen_layers:
                self.t[idx] += 1
                param[0] -= self.lr * model_params[0]["weight_decay"] * param[0]
                self.mt[idx] = beta_1 * self.mt[idx] + (1. - beta_1) * param[1]
                self.vt[idx] = beta_2 * self.vt[idx] + (1. - beta_2) * cp.square(param[1])
                mt_hat = self.mt[idx] / (1 - math.pow(beta_1, self.t[idx]))
                vt_hat = self.vt[idx] / (1 - math.pow(beta_2, self.t[idx]))
                param[0] -= mt_hat * (self.lr / (cp.sqrt(vt_hat) + self.eps))
            idx += 1

        for param in model_params[1]["params"]:
            if param[2] not in self.frozen_layers:
                self.t[idx] += 1
                self.mt[idx] = beta_1 * self.mt[idx] + (1. - beta_1) * param[1]
                self.vt[idx] = beta_2 * self.vt[idx] + (1. - beta_2) * cp.square(param[1])
                mt_hat = self.mt[idx] / (1 - math.pow(beta_1, self.t[idx]))
                vt_hat = self.vt[idx] / (1 - math.pow(beta_2, self.t[idx]))
                param[0] -= mt_hat * (self.lr / (cp.sqrt(vt_hat) + self.eps))
            idx += 1

    def zero_grad(self):
        model_params = self.model.named_parameter_optim_groups(self.weight_decay)

        for param in model_params[0]["params"]:
            param[1].fill(0.)

        for param in model_params[1]["params"]:
            param[1].fill(0.)

    def load_state_dict(self, checkpoint):
        for i in range(len(self.t)):
            self.mt[i] = cp.copy(checkpoint["mt"][i])
            self.vt[i] = cp.copy(checkpoint["vt"][i])
            self.t[i] = checkpoint["t"][i]

    def state_dict(self):
        params = {"mt": self.mt, "vt": self.vt, "t": self.t}
        return params


class GPT:

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = dict(
            wte=Embedding(config.vocab_size, config.n_embd),
            wpe=Embedding(config.block_size, config.n_embd),
            drop=Dropout(config.dropout),
            h=[Block(config) for _ in range(config.n_layer)],
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        self.cross_entropy = CrossEntropy(config.axis)
        self.sm = Softmax(config.axis)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False, dtype=config.dtype)

        self.transformer["wte"].weight = self.lm_head.W.copy() # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply_params(self._init_weights)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        self.loss = None
        self.idx = None
        self.gradient_accumulation_steps = None

    def apply_params(self, fn):
        fn(self.transformer["wte"])
        fn(self.transformer["wpe"])
        for h_block in self.transformer["h"]:
            fn(h_block.attn.c_attn)
            fn(h_block.mlp.c_fc)
            # apply special scaled init to the residual projections, per GPT-2 paper
            h_block.attn.c_proj.W = cp.random.normal(0.0, 0.02/math.sqrt(2 * self.config.n_layer), size=h_block.attn.c_proj.W.shape).astype(dtype=self.config.dtype)
            h_block.mlp.c_proj.W = cp.random.normal(0.0, 0.02/math.sqrt(2 * self.config.n_layer), size=h_block.mlp.c_proj.W.shape).astype(dtype=self.config.dtype)
        fn(self.lm_head)

    def _init_weights(self, module):
        if isinstance(module, Linear):
            module.W = cp.random.normal(0.0, 0.02, size=module.W.shape).astype(dtype=self.config.dtype)
            if module.b is not None:
                module.b = cp.zeros(module.b.shape).astype(dtype=self.config.dtype)
        elif isinstance(module, Embedding):
            module.weight = cp.random.normal(0.0, 0.02, size=module.weight.shape).astype(dtype=self.config.dtype)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        optim_groups = self.named_parameter_optim_groups(0.0)

        num_decay_params = sum(p[0].size for p in optim_groups[0]["params"])
        num_nodecay_params = sum(p[0].size for p in optim_groups[1]["params"])

        total = num_decay_params + num_nodecay_params

        if non_embedding:
            total -= self.transformer["wte"].weight.size

        return total

    def get_param_norms(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        optim_groups = self.named_parameter_optim_groups(0.0)

        num_decay_params = [cp.linalg.norm(p[0]) for p in optim_groups[0]["params"]]
        num_nodecay_params = [cp.linalg.norm(p[0]) for p in optim_groups[1]["params"]]

        param_norms = num_decay_params + num_nodecay_params

        return param_norms

    def forward(self, idx, targets=None, gradient_accumulation_steps=1, calculate_norms = True):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.idx = idx
        b, t = idx.shape[0], idx.shape[1]
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = cp.arange(0, t, dtype=cp.longlong) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer["wte"].forward(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer["wpe"].forward(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        x = self.transformer["drop"].forward(x)
        for block in self.transformer["h"]:
            x = block.forward(x)
        x = self.transformer["ln_f"].forward(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head.forward(x)
            loss = self.cross_entropy.forward(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1) / gradient_accumulation_steps
        else:
            # TODO: Check
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head.forward(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        norms = None
        if calculate_norms:
            norms = sum(self.get_param_norms())

        return logits, loss, norms

    def backward(self, grad=None):
        grad = self.cross_entropy.backward() / self.gradient_accumulation_steps # logits grad
        grad = grad.reshape(self.idx.shape[0], self.idx.shape[1], self.config.vocab_size)
        grad = self.lm_head.backward(grad)
        grad = self.transformer["ln_f"].backward(grad)
        for h_block in reversed(self.transformer["h"]):
            grad = h_block.backward(grad)
        grad = self.transformer["drop"].backward(grad)
        self.transformer["wte"].backward(grad)
        self.transformer["wpe"].backward(grad)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model

        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer["wpe"].weight = self.transformer["wpe"].weight[:block_size]
        for block in self.transformer["h"]:
            block.attn.c_attn.b = block.attn.c_attn.b[:, :, :block_size, :block_size]
            block.attn.c_proj.b = block.attn.c_proj.b[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                sd[k].copy_(sd_hf[k])

        return model

    def named_parameter_optim_groups(self, weight_decay):
        decay_params = []
        nodecay_params = []

        decay_params.append([self.transformer["wte"].weight, self.transformer["wte"].weight_g, "wte.w"])
        decay_params.append([self.transformer["wpe"].weight, self.transformer["wpe"].weight_g, "wpe.w"])
        for i, h_block in enumerate(self.transformer["h"]):
            nodecay_params.append([h_block.ln_1.weight, h_block.ln_1.weight_g, f"block_{i}.ln_1.w"])
            nodecay_params.append([h_block.ln_1.bias, h_block.ln_1.bias_g, f"block_{i}.ln_1.b"])

            decay_params.append([h_block.attn.c_attn.W, h_block.attn.c_attn.Wg, f"block_{i}.attn.c_attn.w"])
            nodecay_params.append([h_block.attn.c_attn.b, h_block.attn.c_attn.bg, f"block_{i}.attn.c_attn.b"])
            decay_params.append([h_block.attn.c_proj.W, h_block.attn.c_proj.Wg, f"block_{i}.attn.c_proj.w"])
            nodecay_params.append([h_block.attn.c_proj.b, h_block.attn.c_proj.bg, f"block_{i}.attn.c_proj.b"])

            nodecay_params.append([h_block.ln_2.weight, h_block.ln_2.weight_g, f"block_{i}.ln_2.w"])
            nodecay_params.append([h_block.ln_2.bias, h_block.ln_2.bias_g, f"block_{i}.ln_2.b"])

            decay_params.append([h_block.mlp.c_fc.W, h_block.mlp.c_fc.Wg, f"block_{i}.mlp.c_fc.w"])
            nodecay_params.append([h_block.mlp.c_fc.b, h_block.mlp.c_fc.bg, f"block_{i}.mlp.c_fc.b"])
            decay_params.append([h_block.mlp.c_proj.W, h_block.mlp.c_proj.Wg, f"block_{i}.mlp.c_proj.w"])
            nodecay_params.append([h_block.mlp.c_proj.b, h_block.mlp.c_proj.bg, f"block_{i}.mlp.c_proj.b"])

        nodecay_params.append([self.transformer["ln_f"].weight, self.transformer["ln_f"].weight_g, f"ln_f.w"])
        nodecay_params.append([self.transformer["ln_f"].bias, self.transformer["ln_f"].bias_g, f"ln_f.b"])

        decay_params.append([self.lm_head.W, self.lm_head.Wg, f"lm_head.w"])
        nodecay_params.append([self.lm_head.b, self.lm_head.bg, f"lm_head.b"])

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        return optim_groups

    def get_params_dict(self):
        model_params = self.named_parameter_optim_groups(0.0)
        model_params = model_params[0]["params"] + model_params[1]["params"]
        params = {}
        for p in model_params:
            params[p[2]] = cp.copy(p[0])
        return params

    def load_from_dict(self, model_dict):
        model_params = self.named_parameter_optim_groups(0.0)
        model_params = model_params[0]["params"] + model_params[1]["params"]
        for p in model_params:
            p[0][True] = cp.copy(model_dict[p[2]])  # True placed for change existing array, not replacing array in list.

    def get_number_of_tensors(self):
        optim_groups = self.named_parameter_optim_groups(None)

        return len(optim_groups[0]['params']) + len(optim_groups[1]['params'])

    def clip_gradient(self, max_norm, clip_norm):
        optim_groups = self.named_parameter_optim_groups(0.0)
        params = optim_groups[0]['params'] + optim_groups[1]['params']
        norms = []
        for p in params:
            norms.append(cp.linalg.norm(p[1], ord=clip_norm))

        total_norm = cp.linalg.norm(
            cp.stack(norms), ord=clip_norm
        )

        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = cp.clip(clip_coef, a_min=1.0, a_max=1.0)

        for p in params:
            p[1] *= clip_coef_clamped

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type=None):
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        optim_groups = self.named_parameter_optim_groups(weight_decay)

        num_decay_params = sum(p[0].size for p in optim_groups[0]["params"])
        num_nodecay_params = sum(p[0].size for p in optim_groups[1]["params"])

        print(f"num decayed parameter tensors: {len(optim_groups[0]['params'])}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(optim_groups[1]['params'])}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        optimizer = AdamW(self, lr=learning_rate, betas=betas, weight_decay=weight_decay)

        return optimizer, optim_groups

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu



    def generate(self, idx: np.ndarray, max_new_tokens, temperature=1.0, top_k=None)-> np.ndarray:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in tqdm(range(max_new_tokens)):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self.forward(cp.asarray(idx_cond))
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                #TODO: Fix for several batches
                v = logits[0][np.argpartition(-logits,top_k)[0]]
                logits[logits < v[:top_k].min()] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = cp.asnumpy(self.sm.forward(logits))
            # sample from the distribution
            idx_next = np.random.default_rng().multinomial(10000, probs).argmax(axis=-1, keepdims=True)
            # append sampled index to the running sequence and continue
            idx = np.concatenate((idx, idx_next), axis=-1)
        return idx

