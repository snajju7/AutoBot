import json
import pickle
import tkinter
import webbrowser
from tkinter import *

import numpy as np
from tensorflow import keras

f = open('01_intents.json', mode='r')
data = json.load(f)

model = keras.models.load_model('01_AutoBot_trained.h5')

with open('01_tokenizer.pkl', mode='rb') as t:
    tokenizer = pickle.load(t)

with open('01_encoder.pkl', mode='rb') as e:
    encoder = pickle.load(e)

wn = tkinter.Tk()
wn.geometry("500x500")
wn.title('AutoBot')
wn.resizable(width=FALSE, height=FALSE)

chat_log = Text(wn, font=('Cambria', 12))
chat_log.config(state=DISABLED)
chat_log.place(x=0, y=0, width=480, height=370)

chat_area = Text(wn, font=('Cambria', 12))
chat_area.place(x=0, y=400, width=400, height=100)


def send():
    user_input = chat_area.get('1.0', 'end-1c')
    chat_area.delete(index1='0.0', index2=END)

    if user_input != "":

        predictions = model.predict(
            keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                                       truncating='post', maxlen=20))

        print(predictions)

        res_index = np.argmax(predictions)
        print(res_index)

        tag = encoder.inverse_transform([res_index])

        print(tag)

        if res_index >= 0:

            for i in data['intents']:

                if i['tag'] == tag:
                    if tag in ['greetings', 'Bye', 'thank_you', 'Good_morning', 'wrong_answer', 'casual_greetings',
                               'casual_greetings_2']:
                        res = np.random.choice(i['response'])
                        chat_log.config(state=NORMAL)
                        chat_log.insert(END, 'You: ' + user_input + '\n\n')
                        chat_log.insert(END, 'AutoBot: ' + str(res) + '\n\n')
                        chat_log.config(state=DISABLED)
                        chat_log.update()

                    else:

                        if i['tag'] == tag:
                            res = np.random.choice(i['response'])
                            res = str(res)
                            assignment_group = i['assignment_group']
                            url = i['url']

                            def open_url():
                                for u in url:
                                    webbrowser.open_new(u)

                            # open_url = lambda a: [webbrowser.open_new(a) for a in url]

                            chat_log.config(state=NORMAL)
                            chat_log.insert(END, 'You: ' + user_input + '\n\n')
                            chat_log.insert(END, 'AutoBot: ' + str(res) + '\n\n')
                            chat_log.insert(END, 'recommended Workgroup is: ' + str(assignment_group) + '\n\n')
                            chat_log.config(state=DISABLED)
                            chat_log.yview(END)

                            for u in url:
                                link_btn = Button(wn, text="Click here to open KB", command=open_url, fg="Blue")
                                link_btn.place(x=0, y=370, width=500, height=31)


        else:
            chat_log.config(state=NORMAL)
            chat_log.insert(END, 'You: ' + user_input + '\n\n')
            chat_log.insert(END, "AutoBot: " + "sorry, I think I don't understand that" + "\n\n")
            # chat_log.insert(END, 'recommended workgroup is: ' + str(assignment_group) + '\n\n')
            chat_log.config(state=DISABLED)


def Enter_key(obj):
    send()


wn.bind('<Return>', func=Enter_key)

send_btn = Button(wn, text='Send', command=send, bg='Orange', fg="Black", font=('Cambria', 14))
send_btn.place(x=400, y=400, width=100, height=100)

scroll_bar = Scrollbar(wn, command=chat_log.yview)
scroll_bar.place(x=480, y=10, width=20, height=390)
wn.mainloop()
