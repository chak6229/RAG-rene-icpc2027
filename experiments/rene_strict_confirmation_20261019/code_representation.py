"""Historical token anonymization with an explicit, execution-mode-independent keep set."""
from __future__ import annotations

import builtins
import keyword


def anonymized_code(tokens):
    declared = {tokens[i + 1] for i, t in enumerate(tokens[:-1]) if t == 'def'}
    keep = set(keyword.kwlist) | {'self', 'cls', 'True', 'False', 'None'} | set(dir(builtins))
    mapping, result = {}, []
    for i, t in enumerate(tokens):
        previous = tokens[i - 1] if i else ''
        following = tokens[i + 1] if i + 1 < len(tokens) else ''
        if previous == '.':
            result.append(t)
        elif t in declared:
            result.append('Func')
        elif t.isidentifier() and t not in keep and following != '(':
            if t not in mapping:
                mapping[t] = f'arg_{len(mapping)}'
            result.append(mapping[t])
        else:
            result.append(t)
    return ' '.join(result)
