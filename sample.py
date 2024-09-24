"""
Sample from a trained model
"""
import os
import pickle
import tiktoken
import utils
from model_gpu import GPTConfig, GPT
from contextlib import nullcontext

import numpy as np


# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "Dear friend. I want" #"\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed_offset = 0
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

np.random.seed(42 + seed_offset)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.npy')
    checkpoint = utils.load_params_dict_cupy(ckpt_path)
    print(checkpoint['config'])
    model_args = dict()
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
        model_args[k] = checkpoint['config'][k]
    #TODO: Fix none vocab size
    model_args['vocab_size'] = 50304 # checkpoint.any().get('config').get('meta_vocab_size')
    gptconf = GPTConfig(**model_args)

    model = GPT(gptconf)
    state_dict = checkpoint['model']
    model.load_from_dict(state_dict)
    del ckpt_path
    del state_dict

elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
    del checkpoint
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (np.array(start_ids, dtype=np.longlong)[None, ...])

# run generation
for k in range(num_samples):
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    print(decode(y[0].tolist()))
    print('---------------')