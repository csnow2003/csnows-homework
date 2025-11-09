
from datasets import load_dataset
from collections import Counter
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
import re

from .utils import SPECIAL_TOKENS

def simple_tokenize(text: str) -> List[str]:
    # Basic cleanup + whitespace split
    text = text.strip().lower()
    # keep apostrophes and hyphens, split on others
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]+(?:'[a-zA-ZÀ-ÿ]+)?|-|[\d]+|[^\s\w]", text)
    if not tokens:
        tokens = [""]
    return tokens

class Vocab:
    def __init__(self, max_size: int = 20000, min_freq: int = 1):
        self.itos = [SPECIAL_TOKENS["pad"], SPECIAL_TOKENS["bos"], SPECIAL_TOKENS["eos"], SPECIAL_TOKENS["unk"]]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.max_size = max_size
        self.min_freq = min_freq

    def build(self, token_lists: List[List[str]]):
        cnt = Counter()
        for lst in token_lists:
            cnt.update(lst)
        # sort by frequency then alphabetically
        sorted_items = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
        for tok, freq in sorted_items:
            if freq < self.min_freq:
                continue
            if tok in self.stoi:
                continue
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)
            if len(self.itos) >= self.max_size:
                break

    def __len__(self):
        return len(self.itos)

    def encode(self, toks: List[str]) -> List[int]:
        unk_id = self.stoi[SPECIAL_TOKENS["unk"]]
        return [self.stoi.get(t, unk_id) for t in toks]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]

class TranslationDataset(Dataset):
    def __init__(self, split: str, max_len: int, max_vocab: int = 20000, cache_vocabs: Dict = None):
        """
        split: 'train' or 'validation' (we'll create our own small val from train)
        """
        self.ds = load_dataset("opus_books", "en-fr")
        data = self.ds["train"]
        # Create small validation split from head/tail
        n = len(data)
        val_count = int(0.02 * n)  # 2% for val
        if split == "train":
            self.samples = [data[i] for i in range(val_count, n)]
        else:
            self.samples = [data[i] for i in range(val_count)]

        self.max_len = max_len
        self.cache_vocabs = cache_vocabs
        if cache_vocabs is not None and "src_vocab" in cache_vocabs:
            self.src_vocab = cache_vocabs["src_vocab"]
            self.tgt_vocab = cache_vocabs["tgt_vocab"]
        else:
            # build vocabs from train portion only
            tokenized_en = []
            tokenized_fr = []
            for i in range(val_count, n):
                tr = data[i]["translation"]
                tokenized_en.append(simple_tokenize(tr["en"]))
                tokenized_fr.append(simple_tokenize(tr["fr"]))
            self.src_vocab = Vocab(max_size=max_vocab)
            self.tgt_vocab = Vocab(max_size=max_vocab)
            self.src_vocab.build(tokenized_en)
            self.tgt_vocab.build(tokenized_fr)

    def __len__(self):
        return len(self.samples)

    def _truncate(self, tokens):
        if len(tokens) > self.max_len - 2:
            tokens = tokens[: self.max_len - 2]
        return tokens

    def __getitem__(self, idx: int):
        item = self.samples[idx]["translation"]
        src_toks = self._truncate(simple_tokenize(item["en"]))
        tgt_toks = self._truncate(simple_tokenize(item["fr"]))

        bos = SPECIAL_TOKENS["bos"]
        eos = SPECIAL_TOKENS["eos"]

        # Encoder input
        src_ids = [self.src_vocab.stoi[bos]] + self.src_vocab.encode(src_toks) + [self.src_vocab.stoi[eos]]
        # Decoder input/output
        tgt_in_ids = [self.tgt_vocab.stoi[bos]] + self.tgt_vocab.encode(tgt_toks)
        tgt_out_ids = self.tgt_vocab.encode(tgt_toks) + [self.tgt_vocab.stoi[eos]]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_in_ids": torch.tensor(tgt_in_ids, dtype=torch.long),
            "tgt_out_ids": torch.tensor(tgt_out_ids, dtype=torch.long),
        }

def collate_pad(batch, pad_id_src: int, pad_id_tgt: int):
    from torch.nn.utils.rnn import pad_sequence
    src_ids = [b["src_ids"] for b in batch]
    tgt_in = [b["tgt_in_ids"] for b in batch]
    tgt_out = [b["tgt_out_ids"] for b in batch]

    src_pad = pad_sequence(src_ids, batch_first=True, padding_value=pad_id_src)
    tgt_in_pad = pad_sequence(tgt_in, batch_first=True, padding_value=pad_id_tgt)
    tgt_out_pad = pad_sequence(tgt_out, batch_first=True, padding_value=pad_id_tgt)

    src_len = torch.tensor([len(x) for x in src_ids], dtype=torch.long)
    tgt_len = torch.tensor([len(x) for x in tgt_in], dtype=torch.long)

    return {
        "src_ids": src_pad,
        "tgt_in_ids": tgt_in_pad,
        "tgt_out_ids": tgt_out_pad,
        "src_len": src_len,
        "tgt_len": tgt_len,
    }
