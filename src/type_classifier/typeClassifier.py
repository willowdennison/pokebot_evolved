#interface class for the type classifier module

import type_classifier.typeClassifierForest as t

default_f1 = 'type1forest.pkl'
default_f2 = 'type2forest.pkl'

class TypeClassifier:
    
    def __init__(self, type_1_forest_path=None, type_2_forest_path=None):
        
        if not type_1_forest_path: type_1_forest_path = default_f1
        if not type_2_forest_path: type_2_forest_path = default_f2
        
        self.t1_forest = t.load_forest(type_1_forest_path)
        self.t2_forest = t.load_forest(type_2_forest_path)
        
    def predTypes(self, name):
        #return tuple (type1, type2)
        return t.classifyTypes(self.t1_forest, self.t2_forest, name)
