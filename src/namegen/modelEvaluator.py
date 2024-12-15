#functions to iterate over, evaluate, and generate output from specific models

import matplotlib.pyplot as plt
import os
import random

from namegen import nameGenInterface


def populateDirectoryList(dir_path):
    model_paths = []
    
    for file in os.listdir(dir_path):
        if file.endswith('.pth'):
            file = dir_path + '/' + file
            model_paths.append(file) 
            
    return model_paths    


def initializeModels(model_path_list): #returns list of models, each stored as [filepath, model, loss]
    models = []
    
    for path in model_path_list:
        
        #grabs initialized model and loss, ignores other values
        _, _, _, loss, _, model = nameGenInterface.loadModel(path)
        
        output = [path, model, loss]
        models.append(output)
    
    return models


def plotLoss(initialized_model_list): #plots the epoch vs the loss of every model in the initialized_model_list
    epochs = []
    loss = []
    
    for i in range(len(initialized_model_list)):
        epochs.append(i+1)
        curr_model = initialized_model_list[i]
        loss.append(curr_model[2])
        
    plt.plot(epochs, loss)
    plt.show()
        

def displayOutput(initialized_model_list, index, n_names=10): #execute the genList function for the model at the specified index
    modelInfo = initialized_model_list[index] #load specified model
    #execute genList() for specified model and return result
    return nameGenInterface.genList(n_names=n_names, specific_model=modelInfo[1], show_syllables=True)


def displayMultipleOutputs(initialized_model_list, index_list, n_names=10): #calls the displayOutput() function with each index in the specified list
    output_list = []
    
    for index in index_list: 
        path = initialized_model_list[index][0]
        loss = initialized_model_list[index][2]
        
        print(f'\nModel {path} test, cross entropy loss value: {loss}') 
        out = displayOutput(initialized_model_list, index, n_names)
        
        output_list.append(out)
        
    return output_list


def modelIndexes(initialized_model_list, silent=True): 
#prints the name and index of each model, returns a dictionary with filenames as keys and indexes as values
#when silent=False, will print every value as it is loaded
    catalog = {} #stores output
    
    for i in range(len(initialized_model_list)):
        path = initialized_model_list[i][0]
        if not silent: print(f'Index {i}: {path}')
        catalog[path] = i
        
    return catalog


def randomIndexes(n_indexes, max_value, min_value=1):
    indexes = []
    for i in range(n_indexes):
        indexes.append(random.choice([j for j in range(min_value, max_value)]))
    return indexes


def sortModels(initialized_model_list, key='epoch'): 
#sorts initialized_model_list by key, accepts 'epoch' (default) or 'loss'

    models_to_sort = []
    sorted_models = []
    
    for model_info in initialized_model_list:
        
        filename = model_info[0].split('/')[-1] #grab just filename, not directories
        
        epoch = int(''.join(filename[19:]).split('l')[0]) #cut out first 19 characters, split at 'l' to remove loss values
        model_info[2] = int(model_info[2]) #format loss as int, not str
        
        model_info.append(epoch)
                
        models_to_sort.append(model_info) #store epoch to be able to sort by it
                        
    match key:
        case 'epoch': #sort by epoch, if epochs are equal sort by loss
            models_to_sort.sort(key = lambda model: (model[3], model[2])) 
        case 'loss': #sort by loss, if loss is equal sort by epoch
            models_to_sort.sort(key = lambda model: (model[2], model[3])) 
        case _:
            raise ValueError("Key must be either 'epoch' or 'loss'")

    sorted_models = models_to_sort
    
    for i in range(len(sorted_models)): 
        model = sorted_models[i]
        del(model[3]) #remove epoch value for compatability
        sorted_models[i] = model
        
    return sorted_models


def outputToFile(output, filename):
    with open(filename, 'w') as file:
        count = 400
        for model_sample in output:
            print(count, end=': ', file=file)
            for name in model_sample:
                name = ''.join(name)
                
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
                
                print(name, end=', ', file=file)
                
            count += 1
            print('\n', file=file)


def main(dir_path, indexes_to_display=True, sort=False, silent=True, plot_loss=False): #evaluates, plots the loss values, and displays and returns output from specified indexes
    #in default case True, will use every model in the directory in order
    #sort takes 'epoch' or 'loss' and will use sortModels()

    path_list = populateDirectoryList(dir_path)

    model_list = initializeModels(path_list)

    if sort:
        try:
            model_list = sortModels(model_list, sort)
        except ValueError:
            print("Invalid sort metric. Valid options are 'epoch' and 'loss'")

    if not silent:
        print('\n-----------\n')
        print(modelIndexes(model_list, silent=False)) #print all indexes to console
    print('\n-----------\n')
    
    if indexes_to_display == True:
        all_indexes = [x for x in range(len(model_list))]
        output = displayMultipleOutputs(model_list, all_indexes)
        
    else:
        output = displayMultipleOutputs(model_list, indexes_to_display)
        
    print('\n-----------\n')
    
    if plot_loss: plotLoss(model_list)
    
    return output



indexes = sorted(randomIndexes(n_indexes=20, max_value=500))
spread = [0, 99, 199, 299, 399, 499]
specific_range = [x for x in range(399, 500, 1)]

#outputToFile(main('namegen/weights/12-11 2 hidden layers', indexes_to_display=specific_range, sort='epoch'), filename='namegen/outputs/last100epochsoutput.txt')

#displayMultipleOutputs(sortModels(initializeModels(populateDirectoryList('src/namegen/weights/12-11 2 hidden layers'))), [447], n_names=100)