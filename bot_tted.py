#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:42:09 2023

@author: midou
""" 

import json
import numpy as np
from tensorflow.keras.models import load_model  
import pickle 
import nltk  
from  nltk.stem import WordNetLemmatizer  
import random



lemmtizer = WordNetLemmatizer() 

intents=json.loads(open('intents.json').read()) 


words=pickle.load(open('words.pkl' , 'rb'))  
clasess=pickle.load(open('classes.pkl' , 'rb')) 

mod=load_model('chatbotm.h5') 


def clean_sentence(sentence): 
    sentence_words=nltk.word_tokenize(sentence) 
    sentence_words=[lemmtizer.lemmatize(word) for word in sentence_words] 
    return sentence_words


def bag_of_words (sentence):  
    sentence_words=clean_sentence(sentence)  
    bag=[0]*len(words) 
    for w in sentence_words: 
        for i , word in enumerate(words): 
            if word==w : 
                bag[i]=1 
    return np.array(bag) 



def predict_class(sentence): 
    bow=bag_of_words(sentence) 
    bow = np.array(bow).reshape((-1, (492,)[0]))    
    res=mod.predict(np.array(bow))[0] 
    Erorr_TH= 0.083333333332  
    results=[[i,r] for i,r in enumerate(res) if r>Erorr_TH  ] 
    results.sort(key=lambda x: x[1] , reverse=True) 
    return_list=[] 
    for r in results : 
          return_list.append({"intents": clasess[r[0]] , "propba":str(r[1])}) 
    
    return return_list 

   
def get_response(pred, jsonn): 
    tag=pred[0]["intents"] 
    list_of_intents=jsonn["intents"] 
    for i in list_of_intents: 
        if i['tag']==tag:
            result=random.choice(i["responses"]) 
            break 
    return result



def get_bot_response(sentence , jsonn): 
    pred=predict_class(sentence) 
    result=get_response(pred=pred, jsonn=jsonn) 
    return(result)









