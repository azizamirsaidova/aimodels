import torch
import numpy as np


def sample(logits, sampling, beams=1, temprature = 1, top_k = 0, top_p = 0.0):
  if sampling == 'random':
    pb = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()
    token_ids = np.random.choice(np.arange(pb.shape[-1]), p=pb)
  elif sampling == 'beam_search':
    token_ids = beam_search(logits, beams)
  elif sampling == 'top_k':
    token_ids = top_k_sampling(logits, temprature, top_k, beams)
  elif sampling == 'top_p':
    token_ids = top_p_sampling(logits, temprature, top_p, beams)
  elif sampling == 'greedy':
    pb = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()
    token_ids = np.argmax(pb) 
  
  return token_ids