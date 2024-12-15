import pandas as pd
import numpy as np
import joblib
from sklearn.multioutput import MultiOutputRegressor
from statgen.statgenTraining import encodeName, encodeType

class StatGen:
    
    def __init__(self):
        
        self.model = joblib.load('statgen/models/statgen.pkl')
        
    def predictStats(self, name, types):
        #takes str name and Tuple(str, str) types
        name = pd.DataFrame([encodeName(''.join(name[0:12]))], columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11'])
        
        encoded = []
        for t in types:
            encoded.append(encodeType(t))
        
        types = pd.DataFrame([encoded], columns=['type_0', 'type_1'])
        
        X = name.join(types)
        
        stats = self.model.predict(X)
        
        for i in range(len(stats)):
            stats[i] = np.round(stats[i], 0)
            
        return stats