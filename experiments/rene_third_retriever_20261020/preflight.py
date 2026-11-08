from __future__ import annotations

import importlib.util

import numpy as np

from .common import MODEL, OUT, SOURCES, now, read, record, require, write
from .encoder import encode, load_encoder, token_ids


def official_class():
    path = SOURCES / 'UniXcoder__downstream-tasks__code-search__model.py'
    receipt = read(path.with_name(path.name + '.receipt.json'))
    require(receipt['sha256'] == record(path)['sha256'], 'Official reference wrapper hash mismatch')
    spec = importlib.util.spec_from_file_location('rene_archived_unixcoder_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def reference_encode(model, tokenizer, texts, length, *, code=False):
    import torch
    # Deliberately reproduce the author's fixed padding and direct tokenizer calls.
    sequences = []
    for text in texts:
        tokens = [tokenizer.cls_token, '<encoder-only>', tokenizer.sep_token]
        tokens += tokenizer.tokenize(text)[:length - 4]
        tokens += [tokenizer.sep_token]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        sequences.append(ids + [tokenizer.pad_token_id] * (length - len(ids)))
    ids = torch.tensor(sequences, device=next(model.parameters()).device)
    with torch.inference_mode():
        wrapper = official_class()(model).eval()
        vectors = wrapper(code_inputs=ids) if code else wrapper(nl_inputs=ids)
    return vectors.cpu().numpy(), sequences


def main():
    if (OUT / 'PREFLIGHT.json').exists():
        require(read(OUT / 'PREFLIGHT.json')['passed'], 'Previous preflight failed')
        return
    import torch
    from transformers import RobertaModel
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(202610203)
    model, tokenizer, loading = load_encoder(MODEL)
    mode_id = tokenizer.convert_tokens_to_ids('<encoder-only>')
    codes = ['def f(a,b): if a>b: return a else return b',
             'def f(a,b): if a<b: return a else return b',
             ' '.join(['def repeated_function ( value ) : return value'] * 70)]
    queries = ['return maximum value', '', 'Python 列表 naïve unicode', ' '.join(['token'] * 300)]
    cv, _, _ = encode(model, tokenizer, codes, 256)
    qv, _, _ = encode(model, tokenizer, queries, 128)
    # A fresh load and the archived author wrapper serve as the reference.
    reference = RobertaModel.from_pretrained(MODEL, local_files_only=True, attn_implementation='eager').float().cuda().eval()
    rc, ci = reference_encode(reference, tokenizer, codes, 256, code=True)
    rq, qi = reference_encode(reference, tokenizer, queries, 128)
    errors = {'code_max_abs': float(np.max(np.abs(cv - rc))), 'query_max_abs': float(np.max(np.abs(qv - rq)))}
    require(max(errors.values()) <= 5e-6, 'Official wrapper parity failed')
    for data, length, padded in [(codes, 256, ci), (queries, 128, qi)]:
        dynamic = token_ids(tokenizer, data, length)
        require(all(a == b[:len(a)] for a, b in zip(dynamic, padded)), 'Official tokenizer parity failed')
    scores = qv[0] @ cv[:2].T
    require(scores[0] > scores[1], 'Official maximum/minimum code-search smoke example failed')
    require(np.allclose(np.linalg.norm(cv, axis=1), 1, atol=2e-6), 'Code unit norms failed')
    require(np.allclose(np.linalg.norm(qv, axis=1), 1, atol=2e-6), 'Query unit norms failed')
    write(OUT / 'PREFLIGHT.json', {'created_utc': now(), 'passed': True, 'qrels_read': False,
          'study_queries_evaluated': False, 'model_revision': '5604afdc964f6c53782a6813140ade5216b99006',
          'mode_id': mode_id, 'pad_id': tokenizer.pad_token_id, 'loading_info': loading,
          'official_wrapper_parity': errors, 'maximum_minimum_cosines': scores.astype(float).tolist(),
          'code_cases': len(codes), 'query_cases': len(queries), 'covers_empty_unicode_truncation': True,
          'reference_is_fresh_actual_model_forward': True,
          'environment': {'torch': torch.__version__, 'device': torch.cuda.get_device_name(0)},
          'file': record(SOURCES / 'UniXcoder__downstream-tasks__code-search__model.py')})
    print({'preflight_passed': True, 'errors': errors, 'example_cosines': scores.tolist()}, flush=True)


if __name__ == '__main__':
    main()
