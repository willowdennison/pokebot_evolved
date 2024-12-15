# base class for accessing the namegen subpackage internally

import namegen.nameGenInterface as ngi
import namegen.modelEvaluator as evaluator

class NameGen:
    def __init__(self, mode='default'):
        self.mode = mode
        
        match mode:
            case 'default':
                best_model, char_to_int, n_vocab, loss, int_to_char, model = ngi.loadModel('namegen/weights/savedmodels/thechosenone-12.11.e448.pth')
            case 'gemini':
                #to be implemented, will interface with gemini api to have it generate the names directly
                pass
            case _: 
                raise ValueError('Invalid mode selection')
        
        self.model = model
        
    def generate(self, prompt=None, n_outputs=1, silent=True, syllables=False):
        return ngi.genList(specific_model=self.model, n_names=n_outputs, prompt=prompt, silent=silent, show_syllables=syllables)