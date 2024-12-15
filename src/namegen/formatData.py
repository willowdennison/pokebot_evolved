#accesses pkmn_dataset and pokemonSyllabized, eliminates some features and combines the data into one master dataset

import pandas as pd
import numpy as np

from namegen import syllabizeNames

syllabizeNames.main() #populate bigDataSyllabized

pkmn_dataset = pd.read_csv('namegen/data/pkmn_dataset.csv')
syllabized = pd.read_csv('namegen/data/bigDataSyllabized.csv')

prunedData = pkmn_dataset.drop(columns=[
        'dex_no', 'generation', 'hatch_time', 'name_fr', 
        'name_es', 'name_de', 'name_it', 'male_percent', 
        'pokedex_colour', 'base_friendship', 'base_experience_yield',
        'ev_yield_hp', 'ev_yield_attack', 'ev_yield_defense',
        'ev_yield_special_attack', 'ev_yield_special_defense',
        'ev_yield_speed', 'description_violet', 'bulbapedia_link'
    ])

mergedData = prunedData.merge(syllabized[['name_en', 'syllables', 'syllable_count']], on='name_en')

mergedData.replace({'stage': {'basic': '1', 'stage 1': '2', 'stage 2': '3'}}, inplace=True)
            
mergedData['base_stat_total'] = mergedData[['base_hp', 'base_atk', 'base_def', 'base_spatk', 'base_spdef', 'base_speed']].sum(axis=1)

print(mergedData.head())
print(mergedData.columns)

#reorder the data
orderedData = mergedData[[
        'name_en', 'syllables', 'syllable_count', 'forms', 'category', 'height_m', 
        'weight_kg', 'has_gender', 'female_percent', 'is_breedable', 'catch_rate', 
        'leveling_rate', 'base_hp', 'base_atk', 'base_def', 'base_spatk', 'base_spdef', 
        'base_speed', 'base_stat_total', 'egg_group_0', 'egg_group_1', 'type_0', 
        'type_1', 'ability_0', 'ability_1', 'hidden_ability', 'is_legendary', 
        'is_mythical', 'is_part_of_evolution', 'stage', 'evolution_level'
    ]]

print(orderedData.head())

orderedData.to_csv('namegen/data/finalData.csv')