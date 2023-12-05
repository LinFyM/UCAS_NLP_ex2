import re
import regex
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

class ProcessData:
    def __init__(self, file_path, vocab_size):
        self.file_path = file_path
        self.vocab_size = vocab_size
        self.text_lines = self.read_text_file()
        self.cleaned_sentences = self.clean_sentences()
        self.word_to_index = self.create_word_to_index()
        self.index_to_word = self.create_index_to_word()

    def read_text_file(self):
        with open(self.file_path, 'r', encoding='gbk') as f:
            text_lines = f.readlines()
        return text_lines

    def clean_sentences(self):
        cleaned_sentences = []
        for line in self.text_lines:
            sentences = re.split(r'/w', line)
            for sentence in sentences:
                cleaned_sentence = re.sub(r'^\d{8}-\d{2}-\d{3}-\d{3}/m', '', sentence)
                cleaned_sentence = re.sub(r'/\w+', '', cleaned_sentence)
                cleaned_sentence = re.sub(r'[\s+\.\!\/_,$%^*(+\"\']+|[+——！“”『 』‘’：；，。？、~@#￥%……&*（）《》\[\]]+', ' ', cleaned_sentence)
                cleaned_sentence = re.sub(r'nt|ns|nz', '', cleaned_sentence)
                if cleaned_sentence != '' and not cleaned_sentence.isspace():
                    cleaned_sentences.append(cleaned_sentence.strip())
        return cleaned_sentences
    
    def split_data(self, test_size=0.2):
        sentences_train, sentences_test = train_test_split(self.cleaned_sentences, test_size=test_size, random_state=42)
        return sentences_train, sentences_test

    def create_word_to_index(self):
        word_counts = Counter(word for sentence in self.cleaned_sentences for word in sentence.split())
        word_to_index_size = self.vocab_size
        most_common_words = word_counts.most_common(word_to_index_size - 1)
        word_to_index = {word: index for index, (word, _) in enumerate(most_common_words)}
        word_to_index["<UNK>"] = word_to_index_size - 1
        return word_to_index
    
    def create_index_to_word(self):
        index_to_word = {word: index for word, index in self.word_to_index.items()}
        return index_to_word

class TextDataset(Dataset):
    def __init__(self, sentences, word_to_index, window_size):
        self.sentences = sentences
        self.word_to_index = word_to_index
        self.window_size = window_size
        self.data = []
        for sentence in self.sentences:
            indexed = [self.word_to_index.get(word, self.word_to_index["<UNK>"]) for word in sentence.split()]
            if len(indexed) < window_size:
                continue
            for i in range(window_size, len(indexed)):
                input = indexed[i - window_size: i]
                target = indexed[i]
                self.data.append((input, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = torch.tensor(self.data[idx][0])
        target = torch.tensor(self.data[idx][1])
        return input, target
    
    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return (inputs, targets)
    
def save_pretrained(index_to_word, embeds, save_path):
    with open(save_path, "w") as writer:
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(index_to_word):
            vec = " ".join(["{:.4f}".format(x) for x in embeds[idx]])
            writer.write(f"{token} {vec}\n")
    print(f"Pretrained embeddings saved to:{save_path}")

# # 打印前十个清洗后的句子
# data_processed = ProcessData(file_path='Annex_02_PKU_Chinese_Corpus199801\ChineseCorpus199801.txt', vocab_size=1000)
# print(data_processed.cleaned_sentences[:10])
