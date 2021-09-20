import torch
# import torchtext
import torch.nn as nn
# from torchtext.vocab import Vocab, build_vocab_from_iterator
# from torchtext.utils import unicode_csv_reader
# from torchtext.data.datasets_utils import _RawTextIterableDataset
from torch import Tensor
from typing import Iterable, List
# import sentencepiece as spm
# import io
import math
import vocab

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# 特殊トークンの定義
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>', '<blk>', '</blk>', '<sep>']

MAX_LEN=80
# sp = spm.SentencePieceProcessor(model_file='corpus_Python-JPN/p3/p3.model')

# def jpn_tokenizer(text):
#   ss = [tok.replace('▁', '') for tok in sp.encode(text, out_type=str)][:MAX_LEN]
#   return [s for s in ss if len(s) != 0]

# def py_tokenizer(text):
#   return [tok for tok in text.split()][:MAX_LEN]

from torch.nn.utils.rnn import pad_sequence

# 連続した操作をまとめて行うためのヘルパー関数
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# SOS/EOSトークンを追加し、入力配列のインデックス用のテンソルを作成
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([SOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

## Transformer の定義

from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)

class PositionalEncoding(nn.Module):
    def __init__(self, 
                 emb_size: int, 
                 dropout: float, 
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + 
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, 
                 num_encoder_layers: int, 
                 num_decoder_layers: int,
                 emb_size: int, 
                 nhead: int, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512, 
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, 
                src: Tensor, 
                tgt: Tensor, 
                src_mask: Tensor,
                tgt_mask: Tensor, 
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, 
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### Masking
## 異なるマスク処理を行う2つの関数を定義

# モデルが予測を行う際に、未来の単語を見ないようにするためのマスク
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# ソースとターゲットのパディングトークンを隠すためのマスク
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def greedy_decode(model, src, src_mask, max_len, beamsize, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])   # prob.size() の実行結果 : torch.Size([1, 1088]) => 1088 はTGT のVOCAV_SIZE
        next_prob, next_word = prob.topk(k=beamsize, dim=1)
        # print(next_word)
        # print(next_prob)

        next_word = next_word[:, 0]     # greedy なので、もっとも確率が高いものを選ぶ
        next_word = next_word.item()   # 要素の値を取得 (int に変換)

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


class NMT(object):
    src_vocab: object
    tgt_vocab: object

    def __init__(self, src_vocab='kujira', tgt_vocab='python'):
        self.src_vocab = vocab.load_vocab(src_vocab)
        self.tgt_vocab = vocab.load_vocab(tgt_vocab)
        tokenizer = vocab.tokenizer_from_vocab(self.src_vocab)
        self.src_transform = sequential_transforms(tokenizer, #Tokenization
                                               self.src_vocab, #Numericalization
                                               tensor_transform) # Add SOS/EOS and create tensor

        # パラメータの定義
        self.SRC_VOCAB_SIZE = len(self.src_vocab)
        self.TGT_VOCAB_SIZE = len(self.tgt_vocab)
        self.EMB_SIZE = 512  # BERT の次元に揃えれば良いよ
        self.NHEAD = 8
        self.FFN_HID_DIM = 512
        self.BATCH_SIZE = 128
        self.NUM_ENCODER_LAYERS = 3
        self.NUM_DECODER_LAYERS = 3

        # インスタンスの作成
        self.transformer = Seq2SeqTransformer(self.NUM_ENCODER_LAYERS, self.NUM_DECODER_LAYERS, 
                                        self.EMB_SIZE, self.NHEAD, self.SRC_VOCAB_SIZE, self.TGT_VOCAB_SIZE,
                                        self.FFN_HID_DIM)

        # TODO: ?
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # デバイスの設定
        self.transformer = self.transformer.to(DEVICE)

        # 損失関数の定義 (クロスエントロピー)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        # オプティマイザの定義 (Adam)
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    def load(self, filename='all-model.pt'):
        self.transformer.load_state_dict(torch.load(filename, map_location=DEVICE))        

    def translate(self, src_sentence: str):
        self.transformer.eval()
        src = self.src_transform(src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(
            self.transformer,  src, src_mask, max_len=num_tokens + 5, beamsize=5, start_symbol=SOS_IDX).flatten()
        return " ".join(self.tgt_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<sos>", "").replace("<eos>", "")

    def translate_beam(self, src_sentence: str):
        """
        複数の翻訳候補をリストで返す。
        """
        ss = []
        ## 
        return ss


def PyNMT(model='model.pt', src_vocab='japanese.pt', tgt_vocab='python.pt'):
    nmt = NMT(src_vocab, tgt_vocab)
    nmt.load(model)
    return nmt


if __name__ == '__main__':
    nmt = NMT()
    nmt.load('./all-model.pt')
    pred = nmt.translate('もし<A>が偶数のとき')
    print('pred:', pred)