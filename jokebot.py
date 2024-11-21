import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.load(open("intents.json"))

words = pickle.load(open("words.pkl","rb"))
classes = pickle.load(open("classes.pkl","rb"))

model = load_model("jokebot_model.h5")

def clean_up_sentance(sentance):
    sentance_words = nltk.word_tokenize(sentance)
    sentance_words = [lemmatizer.lemmatize(word) for word in sentance_words]
    return sentance_words

def bag_of_words(sentance):
    sentance_words = clean_up_sentance(sentance)
    bag = [0] * len(words)
    for w in sentance_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentance):
    bow = bag_of_words(sentance)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD= 0.25
    result = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({"intent":classes[r[0]], "probabilty":str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

print("Greetings, my name is Arthur")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints,intents)
    print(res)



