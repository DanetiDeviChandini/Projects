#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, KFold
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Replace commas with spaces
    text = re.sub(r',', ' ', text)
    
    # Replace parentheses with spaces
    text = text.replace('(', ' ').replace(')', ' ')
    
    # Replace special characters with spaces
    text = re.sub('[^a-zA-Z0-9 \n\.]', ' ', text)

    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove single quotations using regular expressions
    text = re.sub(r"''", " ", text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def filter_skills(text, skills):
    """This functio will extract the set of skills from the text passed
        Input: text string and list of skills
        Output: text string with only skills from the list discarding all other words"""
    
    tokens = word_tokenize(text)
    
    tokens = [token for token in tokens if token in skills]
    
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def keyword_replacement(text):
    text = text.replace('web technology', 'web_technology')
    text = text.replace('visual studio', 'visual_studio')
    text = text.replace('operating system', 'operating_system')
    text = text.replace('sql server', 'sql_server')
    text = text.replace('react j', 'react_js')
    text = text.replace('spring boot', 'spring_boot')
    text = text.replace('node j', 'node_js')
    text = text.replace('window xp', 'window_xp')
    text = text.replace('designing website', 'website_designing')
    text = text.replace('programming language', 'programming_language')
    text = text.replace('responsive design', 'responsive_design')
    text = text.replace('sql server', 'sql_server')
    text = text.replace('pl sql', 'pl_sql')
    text = text.replace('management studio', 'management_studio')
    text = text.replace('oracle 11g', 'oracle11g')
    text = text.replace('microsoft technology', 'microsoft_technology')
    text = text.replace('visual studio', 'visual_studio')
    text = text.replace('cloud platform', 'cloud_platform')
    text = text.replace('control flow', 'control_flow')
    text = text.replace('data conversion', 'data_conversion')
    text = text.replace('programming language', 'programming_language')
    text = text.replace('p sql', 'p_sql')
    
    return text

# Function to extract experience from the resume
def expDetails(Text):
    global sent
   
    Text = Text.split()
   
    for i in range(len(Text)-2):
        Text[i].lower()
        
        if Text[i] ==  'years':
            sent =  Text[i-2] + ' ' + Text[i-1] +' ' + Text[i] +' '+ Text[i+1]
            
            sent = re.sub('[^0-9.]', '', sent)
            sent = re.findall(r'\d+(?:\.\d+)?|\w+', sent)
            return (sent[0])




