import pandas as pd

data = pd.read_csv('namegen/data/finalData.csv')

nameSyllableType = data[['name_en', 'syllables', 'syllable_count', 'type_0', 'type_1']]

syllables = []

for index, row in nameSyllableType.iterrows():
    
    s = row['syllables']
    s = s.split(',')
    
    entry = []
    
    for i in range(5):
        if i < len(s):
            entry.append(s[i].translate({ord(c): None for c in '[]"\' '}).lower())
        else:
            entry.append(None)
    
    syllables.append(entry)
    
syllables = pd.DataFrame(data=syllables, columns=['s1', 's2', 's3', 's4', 's5'])
    
nameSyllableType = nameSyllableType.join(syllables).drop('syllables', axis=1)

print(nameSyllableType.head())

nameSyllableType.to_csv('type_classifier/data/type_syllable.csv')