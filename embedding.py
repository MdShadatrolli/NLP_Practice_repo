import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Lambda,Dense
import tensorflow as tf

text_data=[
    "The quick brown fox jumps over the lazy dog",
    "Word embeddings are a powerful tool in natural language processing",
    "CBOW and Skip-gram are two architectures for Word2Vec"
]

#Tokenize the Text
tokenizer=Tokenizer(num_words=None,oov_token="<unk>")
tokenizer.fit_on_texts(text_data)
word_index=tokenizer.word_index
vocab_size=len(word_index)+1
print(word_index)
sequences=tokenizer.texts_to_sequences(text_data)
print(sequences)

#Generate Context-target pairs
window_size=2
contexts=[]
targets=[]

for sequence in sequences:
    for i in range(window_size,len(sequence)-window_size):
        context_words=sequence[i - window_size:i]+sequence[i +1:i+window_size]
        target_word=sequence[i]
        contexts.append(context_words)
        targets.append(target_word)


