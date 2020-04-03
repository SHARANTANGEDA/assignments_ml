import sys
try:
    from stemming.porter2 import stem
except ImportError:
    print ('You need to install the following stemming package:')
    print ('http://pypi.python.org/pypi/stemming/1.0')
    sys.exit(0)


def sentence_to_stem(sentence):
    # remove end of lines
    sentence_flat = sentence.replace('\r', '\n').replace('\n', ' ').lower()
    sentence_flat = ' ' + sentence_flat + ' '
    # special cases (English...)
    sentence_flat = sentence_flat.replace("'m ", " am ")
    sentence_flat = sentence_flat.replace("'re ", " are ")
    sentence_flat = sentence_flat.replace("'ve ", " have ")
    sentence_flat = sentence_flat.replace("'d ", " would ")
    sentence_flat = sentence_flat.replace("'ll ", " will ")
    sentence_flat = sentence_flat.replace(" he's ", " he is ")
    sentence_flat = sentence_flat.replace(" she's ", " she is ")
    sentence_flat = sentence_flat.replace(" it's ", " it is ")
    sentence_flat = sentence_flat.replace(" ain't ", " is not ")
    sentence_flat = sentence_flat.replace("n't ", " not ")
    sentence_flat = sentence_flat.replace("'s ", " ")
    # remove boring punctuation and weird signs
    punctuation = (',', "'", '"', ",", ';', ':', '.', '?', '!', '(', ')',
                   '{', '}', '/', '\\', '_', '|', '-', '@', '#', '*')
    for p in punctuation:
        sentence_flat = sentence_flat.replace(p, '')
    words = filter(lambda x: x.strip() != '', sentence_flat.split(' '))
    
    # print(sentence_flat)
    # stem words
    words = map(lambda x: stem(x), words)
    
    word_set = {'azbsbs'}
    
    for w in words:
        word_set.add(w)
    
    word_set.remove('azbsbs')
    bow = []
    
    for s in word_set:
        bow.append(s)

    return bow

def input_to_bow(sentence):
    if __name__ == '__main__':
    
        #sentence = input("Enter sentence: ")
        for word in sys.argv[2:]:
            sentence += ' ' + word
        sentence = sentence.strip()
    
        # make bag of words
        bow = sentence_to_stem(sentence)
        print(bow)
    
