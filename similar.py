from gensim.models import KeyedVectors
import random

# 加载保存的词向量
model1 = KeyedVectors.load_word2vec_format('fnn_tested_embedding_layer.pth', binary=False)
model2 = KeyedVectors.load_word2vec_format('fnn_ovfitting_embedding_layer.pth', binary=False)
model3 = KeyedVectors.load_word2vec_format('rnn_embedding_layer.pth', binary=False)
model4 = KeyedVectors.load_word2vec_format('rnn_tested_embedding_layer.pth', binary=False)
model5 = KeyedVectors.load_word2vec_format('rnn_ovfitting_embedding_layer.pth', binary=False)
model6 = KeyedVectors.load_word2vec_format('lstm_embedding_layer.pth', binary=False)
model7 = KeyedVectors.load_word2vec_format('lstm_tested_embedding_layer.pth', binary=False)
model8 = KeyedVectors.load_word2vec_format('lstm_ovfitting_embedding_layer.pth', binary=False)

# 找到与给定单词最相似的单词
def find_similar_words(word, model):
    try:
        similar_words = model.most_similar(word, topn=10)
        return [word for word, similarity in similar_words]
    except KeyError:
        return ["Word not in vocabulary"]

# 从模型中随机选择20个词
random_words = random.sample(model2.index_to_key, 12)

# 固定监测点
random_words.append("俄罗斯")
random_words.append("她")
random_words.append("他")
random_words.append("一个")
random_words.append("１２月")
random_words.append("５")
random_words.append("吃")
random_words.append("拿")
random_words.append("北京")

# 对每个随机选取的单词找到最相似的单词
for word in random_words:
    similar_words1 = find_similar_words(word, model1)
    similar_words2 = find_similar_words(word, model2)
    similar_words3 = find_similar_words(word, model3)
    similar_words4 = find_similar_words(word, model4)
    similar_words5 = find_similar_words(word, model5)
    similar_words6 = find_similar_words(word, model6)
    similar_words7 = find_similar_words(word, model7)
    similar_words8 = find_similar_words(word, model8)

    print(f"Similar words to '{word}' in fnn_tested:  {similar_words1}")
    print(f"Similar words to '{word}' in fnn_ovfitting:  {similar_words2}")
    print(f"Similar words to '{word}' in rnn: {similar_words3}")
    print(f"Similar words to '{word}' in rnn_tested:  {similar_words4}")
    print(f"Similar words to '{word}' in rnn_ovfitting:  {similar_words5}")
    print(f"Similar words to '{word}' in lstm:  {similar_words6}")
    print(f"Similar words to '{word}' in lstm_tested:  {similar_words7}")
    print(f"Similar words to '{word}' in lstm_ovfitting:  {similar_words8}")

    print()