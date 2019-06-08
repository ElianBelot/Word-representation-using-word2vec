# ===============[ IMPORTATION DES MODULES ]===============
import re
import multiprocessing
import nltk
import requests
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.manifold as sk
from gensim.models.word2vec import Word2Vec


# ===============[ TÉLÉCHARGEMENT DU CORPUS DE TEXTE ]===============
nltk.download('punkt')
nltk.download('stopwords')


# ===============[ CONVERSION DES PHRASES EN LISTE DE MOTS ]===============
def sentence_to_wordlist(raw):
    clean = re.sub('[^a-zA-Z]', ' ', raw).lower()  # Remplace tout caractère n'étant pas une lettre de A à Z par un espace et met tout en minuscule
    words = clean.split()  # Transforme la string en liste de strings coupées aux espaces
    return words  # Met chaque élément de la liste en minuscule


# ===============[ TRAITEMENT DU CORPUS DE TEXTE ]===============
filepath = 'http://www.gutenberg.org/files/33224/33224-0.txt'
corpus_raw = requests.get(filepath).text  # Une string de 2,644,993 caractères

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)  # Une liste de 18,225 phrases

# Conversion en liste de phrases, chaque phrase étant une liste de strings
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# ===============[ CONFIGURATION DES HYPERPARAMÈTRES ]===============
num_features = 300  # Dimension des vecteurs de mots
num_workers = multiprocessing.cpu_count()  # Nombre de CPUs à mettre à disposition

seed = 1  # Graine aléatoire
context_size = 7  # Distance maximale entre le mot actuel et celui prédit dans une phrase
downsampling = 1e-3  # Seuil à partir duquel les mots à haute fréquence sont aléatoirement downsampled
min_word_count = 3  # Ignore les mots apparaissant au total moins de 3 fois


# ===============[ DÉFINITION DU MODÈLE ]===============
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


# ===============[ ENTRAÎNEMENT DU MODÈLE ]===============
model2vec.train(sentences, total_examples=model2vec.corpus_count, epochs=100)
model2vec.save('vectors/trained_model.w2v')


# ===============[ RÉDUCTION DE LA DIMENSIONNALITÉ ]===============
tsne = sk.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix_2d = tsne.fit_transform(model2vec.wv.vectors)


# ===============[ AFFICHAGE DU CLUSTER ]===============
word_vectors_list = [(word, all_word_vectors_matrix_2d[model2vec.wv.vocab[word].index]) for word in model2vec.wv.vocab]
word_coordinates_list = [(word, coords[0], coords[1]) for word, coords in word_vectors_list]

points = pd.DataFrame(
    word_coordinates_list,
    columns=['word', 'x', 'y']
)

plt.scatter(points['x'], points['y'])
plt.show()
