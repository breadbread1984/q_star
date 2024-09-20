#!/usr/bin/python3

from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, \
LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, SuppressTokensLogitsProcessor

class QuietStar(nn.Module):
  def __init__(self, model_id, config = {"num_thoughts": 2, "thought_length": 12, "lookahead_tokens": 4}):
    super(QuietStar, self).__init__()
    self.config = config
    self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code = True)
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 1) 向tokenizer添加thought start和thought end两个token
    num_added_tokens = self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|startofthought|>","<|endofthought|>"]})
    assert num_added_tokens == 2
    self.start_thought_token_id = self.tokenizer("<|startofthought|>", return_attention_mask = False)["input_ids"][-1]
    self.end_thought_token_id = self.tokenizer("<|endofthought|>", return_attention_mask = False)["input_ids"][-1]
    # 2) LLM的embedding table的权重，添加两个额外token对应的embedding初始向量
    tok_emb = self.model.get_input_embeddings()
    e, d = tok_emb.weight.shape
    with torch.no_grad():
      if self.start_thought_token_id >= e or self.end_thought_token_id >= e:
        new_embedding_count = max(self.start_thought_token_id, self.end_thought_token_id) + 1 - e
        tok_emb.weight = torch.nn.Parameter(torch.cat([tok_emb.weight, torch.zeros((new_embedding_count, d), dtype = tok_emb.weight.dtype, device = next(tok_emb.parameters()).device)]))
      # get initial embedding weights for the new tokens
      ids = self.tokenizer("---", return_attention_mask = False)["input_ids"][-1]
      init_value = tok_emb.weight[ids].detach().mean(dim = 0) # init_value.shape = ()
      tok_emb.weight[self.start_thought_token_id, :] = init_value
      tok_emb.weight[self.end_thought_token_id, :] = init_value
    # set only new embedding vector are trainable
    trainability_mask = torch.zeros_like(tok_emb,weight, device = next(tok_emb.parameters()).device())
    trainability_mask[self.start_thought_token_id] = 100.0
    trainability_mask[self.end_thought_token_id] = 100.0
    tok_emb.weight.register_hook(lambda grad: grad * trainability_mask)
    # 3) LLM的预测头全连接权重，添加两个额外token对应的weight向量
    lm_head = self.model.lm_head
    if lm_head is not None:
      e, d = lm_head.weight.shape
      with torch.no_grad():
        if self.start_thought_token_id >= e or self.end_thought_token_id >= e:
          new_embedding_count = max(self.start_thought_token_id, self.end_thought_token_id) + 1 - e
          lm_head.weight = torch.nn.Parameter(torch.cat([lm_head.weight, torch.zeros((new_embedding_count, d), dtype = lm_head.weight.dtype, device = next(lm_head.parameters()).device)]))
        # get initial weights for the new tokens
        ids = self.tokenizer("---", return_attention_mask = False)["input_ids"][-1]
        init_value = lm_head.weight[ids].detach().mean(dim = 0) # init_value.shape = ()
        lm_head.weight[self.start_thought_token_id, :] = init_value
        lm_head.weight[self.end_thought_token_id, :] = init_value
    # 4) 创建mixing head
    self.mixing_mlp = torch.nn.Sequential(
      nn.Linear(2 * self.model.config.hidden_size, 2 * self.model.config.hidden_size),
      nn.ReLU(),
      nn.Linear(2 * self.model.config.hidden_size, 1),
      nn.Sigmoid()).to(next(self.model.parameters()).device)
  def sample_token(self, input_ids, logits, do_sample: bool = False, temperature: float = 0.7, top_k: float = -1, top_p: float = 1):
    # logits.shape = (batch, vocab_size)
    logits_processor = LogitsProcessorList()
    logits_processor.append(TemperatureLogitsWarper(temperature))
    if top_p != 1: logits_processor.append(TopPLogitsWarper(top_p))
    else: logits_processor.append(TopKLogitsWarper(top_k))
    logits_processor.append(SuppressTokensLogitsProcessor([self.start_thought_token_id, self.end_thought_token_id]))
    logits = logits_processor(input_ids, scores = logits) # logits.shape = (batch, vocab_size)
    if do_sample:
      tokens = torch.distributions.categorical.Categorical(logits = logits).sample().unsqueeze(-1) # tokens.shape = (batch, 1)
    else:
      tokens = torch.argmax(logits, dim = -1, keepdim = True) # tokens.shape = (batch, 1)
    return tokens
  def forward(self,
              input_ids: torch.Tensor,
              attention_mask: Optional[torch.Tensor] = None,
              past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
              do_sample: bool = False,
              temperature: float = 0.7
              top_k: float = -1,
              top_p: float = 1):
    logits, hidden, past_key_values = self.original_forward(input_ids, attention_mask, past_key_values)
    logits_thought, hidden_thought, _ = self.thoughtful_forward(input_ids, attention_mask, past_key_values, do_sample, temperature, top_k, top_p)
    w = self.mixing_head(torch.cat([hidden, hidden_thought], dim = -1)) # w.shape = (batch, 1)
    weighted_logits = w * logits + (1. - w) * logits_thought # weighed_logits.shape = (batch, vocab_size)
    tokens = self.sample_token(input_ids, weighted_logits) # tokens.shape = (batch, 1)
    return tokens
  def original_forward(self,
              input_ids: torch.Tensor,
              attention_mask: Optional[torch.Tensor] = None,
              past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
    res = self.model.forward(input_ids, attention_mask = attention_mask, past_key_values = past_key_values, use_cache = True, return_dict = True)
    logits = res.logits[:,-1,:]
    past_key_values = res.past_key_values
    hidden = res.hidden_states[:,-1,:] # hidden.shape = (batch, hidden_dim)
    return logits, hidden, past_key_values
  def thoughful_forward(self,
              input_ids: torch.Tensor,
              attention_mask: Optional[torch.Tensor] = None,
              past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
              do_sample: bool = False,
              temperature: float = 0.7
              top_k: float = -1,
              top_p: float = 1):
    b, l = x.shape
    start_thought_token = torch.full((b, 1), self.start_thought_token_id, device = next(self.model.parameters()).device, dtype = torch.int64)
    # 推理thought第一个token
    input_ids = torch.cat([input_ids, start_thought_token], dim = 1) # x.shape = (batch, seq_len + 1)
    # 推理thought
    for _ in range(self.config['thought_length']):
      if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((b,1), device = next(self.model.parameters()).device, dtype = torch.int64)], dim = 1) # attention_mask.shape = (batch, seq_len + i)
      res = self.model.forward(input_ids, attention_mask = attention_mask, past_key_values = past_key_values, use_cache = True, return_dict = True)
      logits = res.logits[:,-1,:] # logits.shape = (batch, vocab_size)
      past_key_values = res.past_key_values
      next_token = self.sample_token(input_ids, logits, do_sample, temperature, top_k, top_p) # next_token.shape = (batch, 1)
      input_ids = next_token
    # 推理正文token
    end_thought_token = torch.full((b, 1), self.end_thought_token_id, device = next(self.model.parameters()).device, dtype = torch.int64)
    input_ids = torch.cat([input_ids, end_thought_token], dim = 1)
    if attention_mask is not None:
      attention_mask = torch.cat([attention_mask, torch.ones((b,2), device = next(self.model.parameters()).device, dtype = torch.int64)], dim = 1)
    res = self.model.forward(input_ids, attention_mask = attention_mask, past_key_values = past_key_values, use_cache = True, return_dict = True)
    logits = res.logits[:,-1,:] # logits.shape = (batch, vocab_size)
    hidden = res.hidden_states[:,-1,:] # hidden.shape = (batch, hidden_dim)
    return logits, hidden, past_key_values
