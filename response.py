
import json
import pickle
import numpy as np
import tensorflow
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

with open('01_intents.json', mode='r') as f:
    data = json.load(f)

with open('01_tokenizer.pkl', mode='rb') as t:
    tokenizer = pickle.load(t)

with open('01_encoder.pkl', mode='rb') as e:
    encoder = pickle.load(e)



model = keras.models.load_model('01_AutoBot_trained.h5')

def autobot_response():

    user_input = input("chat here")

    if user_input != "":
        print(user_input)

        predictions = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]), truncating="post", maxlen=20))

        print(predictions)

        result_index = np.argmax(predictions)

        tag = encoder.inverse_transform([result_index])

        print(tag)


        #if res_index >= 0:

        for i in data['intents']:
            if i['tag'] == tag:
                result = np.random.choice(i['response'])

                assignment_group = i['assignment_group']

                url = i['url']

                print(result, assignment_group)

                return result



