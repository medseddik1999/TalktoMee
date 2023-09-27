#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 21:10:37 2023

@author: midou
"""

from bot_tted import get_bot_response , intents
import streamlit as st    
import os 
import numpy as np 


#dashboard_location = os.path.dirname(__file__)


st.title("Talk To Me 🧑‍💼 💼" ) 
st.markdown("**This bot represents me, and you can ask it questions about my academic background, experience,  availability, and more.**")




if "messages" not in st.session_state: 
    st.session_state.messages = [] 
    
    
    

for message in st.session_state.messages:
    with st.chat_message(message['role']) :
      st.markdown(message['content']) 
      


prop=st.chat_input() 

if prop: 
    with st.chat_message(name='user'): 
         st.markdown(prop) 
    st.session_state.messages.append({'role':"user" ,'content':prop })     
    
    response=get_bot_response(sentence = prop, jsonn=intents)
    
    
    
    with st.chat_message('assistant') :
         st.markdown(response)   
    
    st.session_state.messages.append({'role':'assistant' ,'content':response })  
    
    
    
#streamlit run streamlit_app.py
