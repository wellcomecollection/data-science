import re
import numpy as np
from bs4 import BeautifulSoup
from fastai.text import Tokenizer


def label_linkable_tokens(article_html, tokenizer=Tokenizer(), label_all=True):
    parsed_html = BeautifulSoup(article_html, 'html.parser')

    link_text = [link.text for link in parsed_html.find_all('a')]
    tokenised_links = tokenizer.process_all(link_text)

    tokenised_text = tokenizer.process_all([parsed_html.text])[0]
    
    target = np.zeros(len(tokenised_text))
    
    for link in tokenised_links:
        start_positions = kmp(tokenised_text, link)
        if label_all:            
            for pos in start_positions:
                target[pos : pos + len(link)] = 1
        elif label_all == False and len(start_positions) > 0:
            pos = start_positions[0]
            target[pos : pos + len(link)] = 1
        else: 
            pass

    return tokenised_text, target


def kmp(sequence, sub):
    """         
    Knuth–Morris–Pratt algorithm, returning the starting position
    of a specified sub within another, larger sequence.
    Often used for string matching.
    """
    partial = [0]
    for i in range(1, len(sub)):
        j = partial[i - 1]
        while j > 0 and sub[j] != sub[i]:
            j = partial[j - 1]
        partial.append(j + 1 if sub[j] == sub[i] else j)

    positions, j = [], 0
    for i in range(len(sequence)):
        while j > 0 and sequence[i] != sub[j]:
            j = partial[j - 1]
        if sequence[i] == sub[j]: j += 1
        if j == len(sub): 
            positions.append(i - (j - 1))
            j = 0

    return positions