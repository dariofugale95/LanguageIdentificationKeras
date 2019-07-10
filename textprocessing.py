import numpy as np
import string
import re 
import json

class TextPreprocessor():
    """ Initialize a TextProcessor. Helpful for building a BoW Model.

    :param text: a text to process
    :param language: "it" by default 

    """
    def __init__(self, text):
        self.text = text

    def to_lower_case(self):
        """ Convert to lower case a text
        
        :param None.

        :return text (lowercased)
        """
        return self.text.lower()
    
    def remove_excess_whitespaces(self):
        """ Fix the problem of excess whitespaces in a text.

        :param None.

        :return text.
        """
        self.text = " ".join(self.text.split())
        return self.text

    def remove_punctuation(self):
        """ Remove all punctuations from text
        :param None.

        :return text without punctuations
        """
        punctuation_set = set(string.punctuation)
        self.text = ''.join(char for char in self.text if char not in punctuation_set)
        return self.text

    def preprocess_text(self):
        """ Perform all preprocessing procedure of a given text

        :param None.

        :return text (preprocessed)
        """
        self.text = self.to_lower_case()
        self.text = self.remove_excess_whitespaces()
        self.text = self.remove_punctuation()
        return self.text