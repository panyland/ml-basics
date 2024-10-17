import random
import numpy as np

vocabulary_file='word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r', encoding='utf-8') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding='utf-8') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])

print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)
    
# Main loop for analogy
while True:
    input_term = input("\nEnter word (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        input_words = input_term.split("-")
        first_vector = W[vocab[input_words[0]], :]
        second_vector = W[vocab[input_words[1]], :]
        test_vector = W[vocab[input_words[2]], :]
        diff_vector = np.subtract(second_vector, first_vector)
        anal_vector = np.add(test_vector, diff_vector)
        distances = {}
        for i in range(vocab_size):
            word = ivocab[i]
            vector =  W[vocab[word],:]
            distances[word] = np.linalg.norm(anal_vector-vector)
        closest = sorted(distances.items(), key=lambda x:x[1])[1:3]
        print("Closest analogies:")
        for word, dist in closest:
            print(f"{word}")
            