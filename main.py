from utils import DataLoader

loader = DataLoader('glove.6B.50d.txt')
word,word_map = loader.read_glove_vecs()
triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, loader.complete_analog(*triad,word_map)))
