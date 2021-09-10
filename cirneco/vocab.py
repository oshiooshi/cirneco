import torch
import torchtext.vocab
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter, OrderedDict

def build_vocab_from_list(tokens=["a", "a", "b", "b", "b"], specials=["<unk>"]):
    counter = Counter(tokens)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    v = torchtext.vocab.vocab(ordered_dict)
    for i,sp in enumerate(specials):
        if sp not in v:
            v.insert_token(sp, i)
    v.set_default_index(0)
    return v

def load_vocab_from_tsv(filename, column=0, tokenizer=lambda s: s.split(), specials=["<unk>"]):
    def yield_tokens():
        with open(filename, encoding = 'utf-8') as f:
            for line in f.readlines:
                yield tokenizer(line.strip().split('\t')[column])
    return build_vocab_from_iterator(yield_tokens, specials=specials)

def save_vocab(vocab_obj, basename='untitled'):
    torch.save(vocab_obj, f'{basename}_vocab.pt')

def load_vocab(basename='untitled'):
    if '.' in basename:
        v = torch.load(basename)
    else:
        v = torch.load(f'{basename}_vocab.pt')
    return v

def tokenizer_from_vocab(vocab):
    d = {}
    for i in range(len(vocab)):
        token = vocab.lookup_token(i)
        c = token[0]
        if c not in d:
            d[c] = []
        d[c].append(token)
    for key in d.keys():
        d[key] = tuple(sorted(d[key], key=lambda x: len(x), reverse=True))
    def tokenizer(s):
        ss = []
        while len(s) > 0:
            for token in d.get(s[0], ()):
                if s.startswith(token):
                    ss.append(token)
                    s = s[len(token):]
                    break
            else:
                s = s[1:]
        return ss
    return tokenizer
