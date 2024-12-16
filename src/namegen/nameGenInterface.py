import torch
from namegen.nameModel import NameModel 
import numpy as np
import pandas as pd
import random
from namegen.syllabizeNames import syllabizeNameList

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
 
seq_len = 20

#default model
model_path = 'namegen/weights/savedmodels/thechosenone-12.11.e448.pth'

#get initial text for list of names for prompting model
names = 'namegen/data/names.csv'
raw_text = open(names, encoding='utf-8').read()
raw_text = raw_text.lower()

#load syllables from dataset
fullData = 'namegen/data/finalData.csv'
fullData = pd.read_csv(fullData)
syllables = fullData['syllables']
firstSyllables = []

for name in syllables:
    name = name.split(',') #separate syllables
    for i in range(len(name)):
        name[i] = name[i].translate({ord(c): None for c in '[]"\''}) #eliminate extra characters from formatting
    firstSyllables.append(name[0].lower()) #store first syllable



def loadModel(path, silent=True): #loads model at specified path. returns best_model, char_to_int, n_vocab, loss, int_to_char, model
    best_model, char_to_int, n_vocab, loss = torch.load(path)
    int_to_char = dict((i, c) for c, i in char_to_int.items())

    #load model
    model = NameModel(n_vocab)
    model.load_state_dict(best_model)
    if not silent: print(f'Model {model_path} loaded, cross entropy loss value: {loss}\n')
    
    return best_model, char_to_int, n_vocab, loss, int_to_char, model


best_model, char_to_int, n_vocab, loss, int_to_char, model = loadModel(model_path)


def generate(inputPrompt=None, silent=True, specific_model=None, include_syllables=False): 
#with no input prompt, will select a random first syllable from the syllables list to use as a base.

    #allows use of user specified models
    if specific_model: 
        charToInt = char_to_int
        vocab = n_vocab
        intToChar = int_to_char
        curr_model = specific_model
        
    else: 
        _, charToInt, vocab, _, intToChar, curr_model = loadModel(model_path)

    #take a random selection of the actual names list as input for the model, with new line for initial prompt
    start = np.random.randint(0, len(raw_text)-seq_len)
    prompt = raw_text[start:start+seq_len] + '\n'
    
    if inputPrompt:
        inputPrompt = inputPrompt.lower()
        prompt = prompt + inputPrompt #tack on user input prompt
    else:
        inputPrompt = random.choice(firstSyllables) #randomly select starting syllable
        inputPrompt = inputPrompt.lower() 
        prompt = prompt + inputPrompt #use syllable as input prompt
        
    pattern = [charToInt[c] for c in prompt] #format prompt as ints for model

    curr_model.eval()
    if not silent: print('Prompt: \n"%s"' % prompt)
    with torch.no_grad():
        for i in range(100):
            # format input array of int into PyTorch tensor
            x = np.reshape(pattern, (1, len(pattern), 1)) / float(vocab)
            x = torch.tensor(x, dtype=torch.float32)
            prediction = curr_model(x) #generate output based on input array
            index = int(prediction.argmax()) #convert to one character

            pattern.append(index) #add generated character to the prompt for the output
            pattern = pattern[1:] #delete 1 character from the beginning of prompt, keeping it at seq_len
            
    #joins the last line of the prompt with the first line of the models prediction to display (hopefully) an individual name
    result = prompt.split('\n')[-1] + (''.join(intToChar[char] for char in pattern)).split('\n')[0]
        
    #join with syllabification
    if include_syllables: result = result + ' - ' + str(syllabizeName(result))
            
    if not silent: print(result)
    return result
  
    
def genList(n_names=5, prompt=None, silent=True, specific_model=None, show_syllables=False):
#if no prompt, starting syllable will be randomly selected from the list of actual pokemon's first syllables
#separated by the NLTK library
    
    if not silent:
        if prompt: print(f'Prompt: {prompt}')
        else: print('No prompt entered, starting syllables will be randomly selected.')
    
    list = []
    for i in range(n_names):
        
        done = False
        while not done:
            
            name = generate(prompt, silent=True, specific_model=specific_model, include_syllables=False)
            name = [x for x in name if x != ''] #remove empty strings generated
            
            if ''.join(name) not in firstSyllables: 
                done = True
                
        list.append(name)
    
    if not silent: print('Generated names:')
    for i in range(len(list)):
        
        name = ''.join(list[i])
        
        words = []
        for word in name.split(' '):
            try: 
                word = word[0].upper() + word[1:]
                words.append(word)
            except IndexError: #if word is 1 character
                word = word.upper()
                words.append(word)
        
        if len(words) == 1:
            name = words[0]
        else:
            name = ''
            for word in words:
                name = name + ' ' + word                
            name = ''.join(name[1:]) #trim leading space
        
        if show_syllables:    
            syllables = str(syllabizeName(name))
            if not silent: print(name + ' - ' + syllables) #print syllables of name
        
        else: 
            if not silent: print(name)
            
        list[i] = name
        
    return list


def testModels(model_path_list, n_names=5, prompt=None): #displays results with the specified parameters for all models in the model_path_list
    
    print('\n----------\n')
    
    for path in model_path_list:
        
        _, _, _, _, _, curr_model = loadModel(path, silent=False)
        
        genList(prompt=prompt, n_names=n_names, silent=True, specific_model=curr_model)
        
        print('\n----------\n')


def syllabizeName(name): #wrapper for the syllabizeNames.syllabizeNameList() function
    return syllabizeNameList([name])[0][1]

    

# genList(prompt=None, n_names=20, silent=True)

# print(generate(silent=True, include_syllables=True))

# testModels(path_list)