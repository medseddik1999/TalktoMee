#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:58:42 2023

@author: midou
"""

import random 
import json 
import nltk  
import pickle 
from nltk.stem import WordNetLemmatizer 
from tensorflow.keras.models import Sequential 
from tensorflow.keras import layers 
from tensorflow.keras.optimizers import SGD 
import numpy as np 




lemmtizer = WordNetLemmatizer() 

intents=json.loads(open('intents.json').read())  

#%%
words=[] 
classes=[] 
documents=[] 
ingore_letters=['?','!','&','//' ,'/' , '@' , '.' , ',' , ';']

for intent in intents["intents"]:  
    for pattern in intent ["samples"]: 
        word_list=nltk.word_tokenize(pattern) 
        words.extend(word_list)
        documents.append( (word_list , intent["tag"]) ) 
        if intent['tag'] not in classes: 
            classes.append(intent['tag'])  
            

#%% 
ignore_letters = set(ingore_letters)

# Use a set comprehension to lemmatize words and filter out those in ignore_letters
words = [lemmtizer.lemmatize(word) for word in words if word not in ignore_letters]


pickle.dump(words, open('words.pkl' , 'wb')) 
pickle.dump(classes, open('classes.pkl' , 'wb'))



training=[] 
outputs_empty=[0]*len(classes)
 

for doc in documents: 
    bag=[]  
    word_patterns=doc[0]
    word_patterns=[lemmtizer.lemmatize(word.lower())for word in word_patterns ]  
    for word in words: 
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_raw=list(outputs_empty) 
    output_raw[classes.index(doc[1])]=1 
    training.append([bag ,output_raw])
    
    

random.shuffle(training) 
training=np.array(training , dtype=list)   

train_x = list(training[:, 0])
train_y = list(training[:, 1])

  

model=Sequential() 
model.add(layers.Dense(200 , input_shape=(340,) , activation='relu' )) 
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(300 , activation='relu' )) 
model.add(layers.Dropout(0.75))  
model.add(layers.Dense(100 , activation='relu' ))  
model.add(layers.Dropout(0.5))  
model.add(layers.Dense(16 , activation='relu' ))  
model.add(layers.Dense(16 , activation='softmax' )) 


model.compile(optimizer='adam',  # You can choose a different optimizer
              loss='categorical_crossentropy',  # Choose an appropriate loss function
              metrics=['accuracy'])   



model.fit(np.array(train_x),np.array(train_y), epochs=200, batch_size=7)  

 
'''
reshaped_array = np.array(train_x[0]).reshape((-1, (525,)[0]))    

model.predict(np.array(reshaped_array ))   

  


[print(i.shape, i.dtype) for i in model.inputs]
[print(o.shape, o.dtype) for o in model.outputs]
[print(l.name, l.input_shape, l.dtype) for l in model.layers]

'''

model.save("chatbotm2.h5") 






