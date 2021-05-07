import nltk
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm as tqdm
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import MWETokenizer



reviews = pd.read_csv(r"C:\Users\ehcho\ner_bert\data\imbd_csv\imdb_train.csv", sep="\t", encoding="UTF-8",
                   names=['pn', 'rv'], header=None)



reviews = reviews.rv
#print(reviews)
#mwe = MWETokenizer(reviews)


def spans(txt):
    tokens = MWETokenizer.tokenize(word_tokenize(txt))
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield token, offset, offset+len(token)
        offset += len(token)


all_item = []

for i in tqdm(range(len(reviews))):
    word_ls = []
    #print("sentence:",i+1)

    for token in spans(reviews.rv[i]):
        print(token)
        # assert token[0]==reviews.location[i][token[1]:token[2]]
        my_tuple = token[0]
        #print("###", token)

        # my_tuples = ' , '.join(map(str, my_tuple))
        if token[0] in reviews:
            # word_ls.append(my_tuple)
            subwords = token[0].split()
            pos_list = [nltk.pos_tag([w]) for w in subwords]
            tag_list = ['I-LOC'] * len(pos_list)
            tag_list[0] = 'B-LOC'

            for s, p, t in zip(subwords, pos_list, tag_list):
                if type(p) == list:
                    p = p[0][1]
                    # print('list')
                new_item = dict({'Sentence #': i + 1, 'Tag': t, 'Word': s, 'POS': p})
                all_item.append(new_item)

            # lis_lo = nltk.pos_tag(word_ls),LOC
            # print(' , '.join(map(str, lis_lo)))

        else:
            print(type(my_tuple))
            my_pos = nltk.pos_tag([my_tuple])[0][1]
            new_item = dict({'Sentence #': i + 1, 'Tag': 'O', 'Word': my_tuple.lower(), 'POS': my_pos})
            all_item.append(new_item)

            # if i+1 == 5177:
            # print(new_item)

            # print(nltk.pos_tag([my_tuple]),',','O')
        # print(new_item)
        # if not(my_pos == '.' or  my_pos == ',' or my_pos == ':' or my_pos == '(' or my_pos == ')') :

        # all_item.append(new_item)

