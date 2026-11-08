from __future__ import annotations

import numpy as np


def token_ids(tokenizer, texts, max_length):
    if max_length < 4:
        raise ValueError('Encoder-only prefix requires four special tokens')
    output = []
    for text in texts:
        tokens = [tokenizer.cls_token, '<encoder-only>', tokenizer.sep_token]
        tokens += tokenizer.tokenize(text)[:max_length - 4]
        tokens += [tokenizer.sep_token]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        if ids[1] == tokenizer.unk_token_id:
            raise ValueError('Missing UniXcoder mode token')
        output.append(ids)
    return output


def load_encoder(model_path, device='cuda'):
    from transformers import RobertaModel, RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
    model, loading = RobertaModel.from_pretrained(model_path, local_files_only=True,
                       output_loading_info=True, attn_implementation='eager')
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise ValueError('Incomplete model loading: ' + str(loading))
    model = model.float().to(device).eval()
    return model, tokenizer, loading


def encode(model, tokenizer, texts, max_length, *, batch_size=16):
    import torch
    device = next(model.parameters()).device
    vectors = []
    lengths = []
    truncated = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            sequences = token_ids(tokenizer, batch, max_length)
            width = max(map(len, sequences))
            input_ids = torch.tensor([s + [tokenizer.pad_token_id] * (width - len(s))
                                      for s in sequences], device=device)
            mask = input_ids.ne(tokenizer.pad_token_id)
            hidden = model(input_ids=input_ids, attention_mask=mask, return_dict=True).last_hidden_state
            pooled = (hidden * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(normalized.cpu().numpy())
            lengths.extend(map(len, sequences))
            truncated.extend(len(tokenizer.tokenize(s)) > max_length - 4 for s in batch)
    return np.concatenate(vectors).astype(np.float32), lengths, truncated


def stable_order(values, doc_ids, depth=100):
    values = np.asarray(values)
    doc_ids = np.asarray(doc_ids)
    if values.shape != doc_ids.shape or len(set(doc_ids.tolist())) != len(doc_ids) or not np.isfinite(values).all():
        raise ValueError('Invalid ranking inputs')
    return np.lexsort((doc_ids, -values))[:depth]
