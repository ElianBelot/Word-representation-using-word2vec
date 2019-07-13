# ===============[ IMPORTS ]===============
import re
import multiprocessing
import nltk
import requests
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.manifold as sk
from gensim.models.word2vec import Word2Vec


# ===============[ LOADING TEXT CORPUS ]===============
nltk.download('punkt')
nltk.download('stopwords')


# ===============[ TEXT PRE-PROCESSING ]===============
def sentence_to_wordlist(raw):
    clean = re.sub('[^a-zA-Z]', ' ', raw).lower()
    words = clean.split()
    return words


# ===============[ DATASET PRE-PROCESSING ]===============
filepath = 'http://www.gutenberg.org/files/33224/33224-0.txt'
corpus_raw = requests.get(filepath).text

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)

sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# ===============[ HYPERPARAMETERS TUNING ]===============
num_features = 300
num_workers = multiprocessing.cpu_count()

seed = 1
context_size = 7
downsampling = 1e-3
min_word_count = 3


# ===============[ MODEL BUILDING ]===============
model2vec = Word2Vec(
    size=num_features,
    workers=num_workers,
    sg=1,
    seed=seed,
    window=context_size,
    sample=downsampling,
    min_count=min_word_count,
)

model2vec.build_vocab(sentences)


# ===============[ TRAINING ]===============
model2vec.train(sentences, total_examples=model2vec.corpus_count, epochs=100)
model2vec.save('vectors/trained_model.w2v')


# ===============[ DIMENSIONALITY REDUCTION ]===============
tsne = sk.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix_2d = tsne.fit_transform(model2vec.wv.vectors)


# ===============[ CLUSTER VISUALIZATION ]===============
word_vectors_list = [(word, all_word_vectors_matrix_2d[model2vec.wv.vocab[word].index]) for word in model2vec.wv.vocab]
word_coordinates_list = [(word, coords[0], coords[1]) for word, coords in word_vectors_list]

points = pd.DataFrame(
    word_coordinates_list,
    columns=['word', 'x', 'y']
)

plt.scatter(points['x'], points['y'])
plt.show()
