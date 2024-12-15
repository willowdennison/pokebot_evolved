#takes the csv files for each generation of pokemon and prints to one csv 
#file where each row is the pokemon's name and the syllable breakdown

import numpy as np
import pandas as pd
from nltk.tokenize.sonority_sequencing import SyllableTokenizer
from nltk.tokenize import word_tokenize

def syllabizeNames(pokemonData): #returns a list of lists
#containing each pokemon's name and the list of tokenized syllables
    result = []
    tokenizer = SyllableTokenizer()
    
    for generation in pokemonData:
        
        names = generation['name_en']
        
        for name in names:
            
            name = word_tokenize(name)
            
            entry = [name] #entry to be added to result list
            
            wordsSyllabized = []
            for word in name: #handle multi word pokemon names
                wordsSyllabized.append(tokenizer.tokenize(word))
            
            syllables = []
            for word in wordsSyllabized: #combine multi word syllable lists
                for syllable in word:
                    if syllable not in ['-', '.', '(', ')']:
                        syllables.append(syllable.lower())
            
            syllables = tokenizer.validate_syllables(syllables)
            
            entry.append(syllables)
            entry.append(len(syllables))
            
            #print(entry)
            
            result.append(entry)
    
    result = fixNames(result)
    
    return result

def fixNames(syllabizedSet, silent=True): #takes the list of names and syllable breakdowns and reformats the names to be single strings
    
    newSet = []
    
    for entry in syllabizedSet:
        newEntry = []
        
        #process names
        name = ''
        wordCount = 0 #track if spaces should be added
        
        for word in entry[0]: #each string in the list currently in the name column
            
            if wordCount > 0: #check if space should be added
                name = name + ' '
                
            name = name + word
            wordCount += 1
        
        newEntry.append(name) #add reformatted name
        newEntry.append(entry[1]) #add syllabized name
        newEntry.append(entry[2]) #add syllable count 
        
        if not silent: print(newEntry)
        
        newSet.append(newEntry)
    
    return newSet

def syllabizeNameList(names): 
#takes a list of names, will return a list containing lists, each containing the given name, a list of syllables, and the number of syllables
    
    result = []
    tokenizer = SyllableTokenizer()
    
    for name in names:
            
            name = word_tokenize(name)
            
            entry = [name] #entry to be added to result list
            
            wordsSyllabized = []
            for word in name: #handle multi word pokemon names
                wordsSyllabized.append(tokenizer.tokenize(word))
            
            syllables = []
            for word in wordsSyllabized: #combine multi word syllable lists
                for syllable in word:
                    if syllable not in ['-', '.', '(', ')']:
                        syllables.append(syllable.lower())
            
            syllables = tokenizer.validate_syllables(syllables)
            
            entry.append(syllables)
            entry.append(len(syllables))
            
            result.append(entry)
            
    result = fixNames(result)
    
    return result

def populateDataset():
    
    # gen1 = pd.read_csv("gen01.csv")
    # gen2 = pd.read_csv("gen02.csv")
    # gen3 = pd.read_csv("gen03.csv")
    # gen4 = pd.read_csv("gen04.csv")
    # gen5 = pd.read_csv("gen05.csv")
    # gen6 = pd.read_csv("gen06.csv")
    # gen7 = pd.read_csv("gen07.csv")
    # gen8 = pd.read_csv("gen08.csv")
    # gen9 = pd.read_csv("gen09.csv")

    bigDataSet = pd.read_csv('namegen/data/pkmn_dataset.csv')

    # allGenData = [gen1, gen2, gen3, gen4, gen5, gen6, gen7, gen8, gen9]

    fullDataSet = [bigDataSet]
    tokenizer = SyllableTokenizer()
    
    syllabizedData = syllabizeNames(fullDataSet, tokenizer)

    df = pd.DataFrame(syllabizedData, columns=['name_en', 'syllables', 'syllable_count'])
    df.to_csv("namegen/data/bigDataSyllabized.csv")
    
#populateDataset()

#print(syllabizeNameList(['Bulbasaur', 'Charmander', 'Squirtle']))