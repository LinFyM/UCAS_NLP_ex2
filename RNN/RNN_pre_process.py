import re
import regex
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random
from sklearn.model_selection import train_test_split

class BucketedDataLoader:
    def __init__(self, data, batch_size, shuffle=True):
        self.buckets = {}
        for item in data:
            seq_len = len(item[0])
            if seq_len not in self.buckets:
                self.buckets[seq_len] = []
            self.buckets[seq_len].append(item)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        for seq_len in self.buckets:
            if self.shuffle:
                random.shuffle(self.buckets[seq_len])
            for i in range(0, len(self.buckets[seq_len]), self.batch_size):
                batch = self.buckets[seq_len][i: i + self.batch_size]
                inputs, targets, lengths = zip(*batch)
                lengths = torch.tensor([len(x) for x in inputs])
                inputs = torch.stack(inputs)
                targets = torch.stack(targets)
                yield inputs, targets, lengths

    def __len__(self):
        total_batches = 0
        for bucket in self.buckets.values():
            total_batches += (len(bucket) + self.batch_size - 1) // self.batch_size
        return total_batches

class ProcessData:
    def __init__(self, file_path):
        self.file_path = file_path
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
        word_to_index_size = 1000
        most_common_words = word_counts.most_common(word_to_index_size - 1)
        word_to_index = {word: index for index, (word, _) in enumerate(most_common_words)}
        word_to_index["<UNK>"] = word_to_index_size - 1
        return word_to_index
    
    def create_index_to_word(self):
        index_to_word = {word: index for word, index in self.word_to_index.items()}
        return index_to_word

class TextDataset(Dataset):
    def __init__(self, sentences, word_to_index):
        self.data = []
        for sentence in sentences:
            indexed = [word_to_index.get(word, word_to_index["<UNK>"]) for word in sentence.split()]
            for i in range(1, len(indexed)):
                input = indexed[: i]
                target = indexed[i]
                self.data.append((torch.tensor(input), torch.tensor(target)))
        self.data.sort(key=lambda x: len(x[0]))

    def __getitem__(self, index):
        input, target = self.data[index]
        return input, target, len(input)

    def __len__(self):
        return len(self.data)
    
def save_pretrained(index_to_word, embeds, save_path):
    with open(save_path, "w") as writer:
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(index_to_word):
            vec = " ".join(["{:.4f}".format(x) for x in embeds[idx]])
            writer.write(f"{token} {vec}\n")
    print(f"Pretrained embeddings saved to:{save_path}")