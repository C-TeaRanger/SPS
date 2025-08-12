
import json

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):

        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]  # 由单词 -> 返回索引

    def __len__(self):
        # 返回词汇表大小
        return len(self.word2idx)


def deserialize_vocab(src):
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d["word2idx"]
    vocab.idx2word = d["idx2word"]
    vocab.idx = d["idx"]
    return vocab