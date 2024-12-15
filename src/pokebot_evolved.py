#the discord bot to integrate with the neural network models, accept prompts, and post its output in discord servers
import pandas as pd
import numpy as np
from namegen.namegen import NameGen
from type_classifier.typeClassifier import TypeClassifier
from statgen.statgen import StatGen
import type_classifier.typeClassifierForest as t

import google.generativeai as genai
import json
import dotenv
import os

#suppress warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"


class PokebotEvolved:
    
    def __init__(self, mode='default', type_forest_paths=[None, None]):
        
        self.mode = mode
        self.namegen = NameGen(mode)
        
        self.tc = TypeClassifier(type_forest_paths[0], type_forest_paths[1])
        
        self.sg = StatGen()
        
        config = dotenv.dotenv_values('gemini_integration/.env')
        api_key = config['API_KEY_SERVICE']
        
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel()
        
        
    def generateName(self, prompt=None, n_names=1, print_to_console=True, show_syllables=False):
        
        silent = not print_to_console
        
        if not silent: print('\n----------\n')
        names = self.namegen.generate(prompt, n_names, silent, show_syllables)
        if not silent: print('\n----------\n')
        
        if n_names == 1: return names[0]
        return names
        
        
    def predictTypes(self, name):
        
        predTypes = self.tc.predTypes(name)
        
        types = []
        for t in predTypes:
            if t == 'nan': t = None
            else: t = t[0].upper() + ''.join(t[1:])
            types.append(t)
            
        return types
    
    
    def predictStats(self, name, types):
        
        stats = self.sg.predictStats(name, types)
        
        stats = pd.DataFrame(stats, columns=['hp', 'atk', 'def', 'spatk', 'spdef', 'speed'])
        
        return stats.astype(int)
    
    
    def genNamesTypes(self, n_names=1, print_to_console=True, prompt=None):
        
        names = []
        
        for n in range(n_names):
            
            name = self.generateName(print_to_console=False, prompt=prompt)
            types = self.predictTypes(name)
            
            if types[1] is None: types = types[0]
            else: types = types[0] + ', ' + types[1]
            
            names.append([name, types])
            
        if print_to_console:
            print('\n----------\n')
            for name in names:
                print(f'Pokemon: {name[0]}')
                print(f'Types: {name[1]}')
                print('\n')
            print('\n----------\n')
        
        if n_names == 1: return names[0]
        return names
    
    
    def nameTypeStat(self, prompt=None, n_to_gen=1, print_to_console=True):
        
        results = []
        
        if print_to_console: print('\n----------\n')
        
        for n in range (n_to_gen):
            
            name = self.generateName(prompt=prompt, print_to_console=False)
            types = self.predictTypes(name)
            stats = self.predictStats(name, types)
            
            if print_to_console:
                print(f'Pokemon: {name}')
                print(f'Types: {types[0]}, {types[1]}')
                print(f'Stats: \n{stats.to_string(index=False)}')
                print('\n')
                
            results.append((name, types, list(np.ravel(stats))))
            
        if print_to_console: print('\n----------\n')
            
        if n_to_gen == 1: return results[0]
        else: return results
        
        
    def generateDescription(self, name, types, stats):
        #configured to take input from nameTypeStat()
        
        prompt = f"""
            Generate a description and Pokedex category for the new {types} type Pokemon {name} in the style of the entries from the Pokemon games. 
            Its base stats are: {stats}. Do not specifically mention these stats in the description, only use them to understand its strengths. 
            Return the description in the following JSON schema:
            """ + """
            {
                'category': str
                'description': str
            }
            """
            
        response = self.gemini_model.generate_content(prompt)
        d = json.loads(''.join(response.text[7:-4]))
        return d['category'], d['description']
            
            
    def generateFullEntry(self, prompt=None, print_to_console=True):
        #generates a new pokemon's name, classifies the type, predicts stats, and uses Gemini to generate a Pokedex description
        #returns dict with keys 'name', 'category', 'type1', 'type2', 'hp', 'atk', 'def', 'spatk', 'spdef', 'speed', 'category', 'description'
        
        nts = self.nameTypeStat(prompt=prompt, print_to_console=False)
        
        description = self.generateDescription(nts[0], nts[1], nts[2])
        
        entry = {
            'name': nts[0],
            'type1': nts[1][0],
            'type2': nts[1][1],
            'hp': nts[2][0],
            'atk': nts[2][1],
            'def': nts[2][2],
            'spatk': nts[2][3],
            'spdef': nts[2][4],
            'speed': nts[2][5],
            'category': description[0],
            'description': description[1]
        }
        
        if print_to_console:
            
            print('\n----------\n')
            print(f'Name: {nts[0]}')
            print(f'Category: {description[0]}')
            
            if nts[1][1]: print(f'Types: {nts[1][0]}, {nts[1][1]}')
            else: print(f'Type: {nts[1][0]}')
            
            print('Stats:')
            stats = pd.DataFrame([nts[2]], columns=['HP', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed'])
            print(stats.to_string(index=False))
            
            print(f'Description: {description[1]}')
            print('\n----------\n')
            
        return entry
    
    
    def generateMultipleEntries(self, n_to_generate=5, prompt=None, print_to_console=True):
        #calls the generateFullEntry function n_to_generate times, returns [dict]
        
        output = []
        
        for n in range(n_to_generate): 
            output.append(self.generateFullEntry(prompt=prompt, print_to_console=print_to_console))
            
        return output
            
        
    def main(self):
        pass
    

    
pb = PokebotEvolved()

pb.generateMultipleEntries()