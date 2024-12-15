import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

from type_classifier.typeUtils import tokenizeText, encodeType, decodeType, encodeName

data_path = 'type_classifier/data/type_syllable.csv'
data = pd.read_csv(data_path)

default_type1_forest_path = 'type_classifier/models/type1forest.pkl'
default_type2_forest_path = 'type_classifier/models/type2forest.pkl'

def train_fit_forest(
    type_to_classify, data=data, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None
    ):
    # trains and fits a random decision forest based on the data as formatted in the namesyllabletype.py formatter:
    # pandas dataframe with columns 'name_en', 'syllable_count', 'type_0', 'type_1', 's1', 's2, 's3', 's4', 's5'
    # type_to_classify takes 1 or 2, and specifies if model should be trained to classify first or second type
    # all other parameters are passed directly to the RandomForestClassifier
    # returns trained and fitted RandomForestClassifier
    
    names = data[['name_en']] 
    
    #store each character of each name after they're encoded
    encoded_names = []
    
    #encode all names to use as tree inputs
    for _, row in names.iterrows():
        encoded_names.append(encodeName(row[0]))
    
    encoded_names = pd.DataFrame(encoded_names, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11'])
        
    match type_to_classify:
        case 1: types = data[['type_0']]; type_to_classify = 'type_0'
        case 2: 
            types = data[['type_1']]
            type_to_classify='type_1'
            
            type1 = np.ravel(data[['type_0']])
            for i in range(len(type1)):
                type1[i] = int(encodeType(type1[i]))
        
            type1 = pd.DataFrame(type1, columns=['type_0'])
            print(type1.head())
            
            encoded_names = encoded_names.join(type1)
            
            print(encoded_names.head())
            
        case _: raise ValueError('type_to_classify must be either 1 or 2')
    
    types = np.ravel(types)
    for i in range(len(types)):
        types[i] = int(encodeType(types[i]))
        
    types = pd.DataFrame(types, columns=[type_to_classify])
    
    forest = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, 
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf, 
        max_features=max_features, max_leaf_nodes=max_leaf_nodes
        )
    
    forest.fit(encoded_names.astype('int'), np.ravel(types).astype('int'))
    
    return forest

def save_forest(forest, filename): 
    # takes a trained and fitted RandomForestClassifier and writes it to the specified filename in type_classifier/models
    filename = 'type_classifier/models/' + filename #add directory
    joblib.dump(forest, filename) #write to file

def load_forest(filename):
    # take a filename, load and return the RandomForestClassifier from type_classifier/models/[filename]
    filename = 'type_classifier/models/' + filename
    forest = joblib.load(filename)
    return forest

def classifyType1(forest, name, silent=True):
    # take a pokemon name and a fitted RandomForestClassifier, predict the type and return as str
    
    if not silent: print(name)
    
    if len(name) > 12:
        name = ''.join(name[0:12])
    
    name = pd.DataFrame([encodeName(name)], columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11'])
    
    if not silent: print(name.head())
    
    pred = forest.predict(name)
    
    if not silent: print(decodeType(pred[0]))
    
    return decodeType(pred[0])

def classifyType2(forest, name, type1, silent=True):
    
    if not silent: print(name)
    
    if len(name) > 12:
        name = ''.join(name[0:12])
        
    name = pd.DataFrame([encodeName(name)], columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11'])
    
    name = name.join(pd.DataFrame([[encodeType(type1)]], columns=['type_0']))
    
    if not silent: print(name.head())
    
    pred = forest.predict(name)
    
    if not silent: print(decodeType(pred[0]))
    
    return decodeType(pred[0])
    
    
def classifyTypes(type1_forest, type2_forest, name):
    # take a pokemon name and a fitted RandomForestClassifier for each type, predict both types and return as Tuple(str, str)
    type1 = classifyType1(type1_forest, name)
    type2 = classifyType2(type2_forest, name, type1)
    return type1, type2


# train_fit_forest(data, 1)