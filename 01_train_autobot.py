import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder


f = open("01_intents.json", mode='r')
data = json.load(f)

training_set = []
training_labels = []
labels = []
response = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_set.append(pattern)
        training_labels.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

    response.append(intent['response'])

vocab_size = 1000
oov_token = '<oov>'
max_len = 20
embedding_dim = 10

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_set)
word_index = tokenizer.word_index
#print(word_index)

seq = tokenizer.texts_to_sequences(training_set)
#print(seq)

padded_seq = pad_sequences(sequences=seq, maxlen=max_len, padding='post')

#print(padded_seq)

encoder = LabelEncoder()
encoder.fit(training_labels)
target = encoder.transform(training_labels)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(padded_seq, np.array(target), epochs=500)

history = model.save('01_AutoBot_trained.h5')

with open('01_tokenizer.pkl', mode='wb') as t:
    pickle.dump(tokenizer, t, pickle.HIGHEST_PROTOCOL)

with open('01_encoder.pkl', mode='wb') as e:
    pickle.dump(encoder, e, pickle.HIGHEST_PROTOCOL)

print('AutoBot trained')








