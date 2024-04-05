import gensim.downloader
import numpy as np
import openai
from scipy.spatial.distance import cdist

word_encoder = gensim.downloader.load('glove-twitter-25')

def embed_sequence(sequence):
    vects = word_encoder[sequence.split(' ')]
    return np.mean(vects, axis=0)


def calc_distance(embedding1, embedding2):
    return cdist(np.expand_dims(embedding1, axis=0), np.expand_dims(embedding2, axis=0), metric='cityblock')[0][0]

def retreive_relevent(prompt, documents):
    min_dist = 1000000000
    r_docname = ""
    r_doc = ""

    for docname, doc in documents.items():
        dist = calc_distance(embed_sequence(prompt)
                           , embed_sequence(doc))
        
        if dist < min_dist:
            min_dist = dist
            r_docname = docname
            r_doc = doc

    return r_docname, r_doc

def retreive_and_agument(prompt, documents):
    docname, doc = retreive_relevent(prompt, documents)
    return f"Answer the customers prompt based on the folowing documents:\n==== document: {docname} ====\n{doc}\n====\n\nprompt: {prompt}\nresponse:"

def main():

    documents = {"menu": "ratatouille is a stew thats twelve dollars and fifty cents also gazpacho is a salad thats thirteen dollars and ninety eight cents also hummus is a dip thats eight dollars and seventy five cents also meat sauce is a pasta dish thats twelve dollars also penne marinera is a pasta dish thats eleven dollars also shrimp and linguini is a pasta dish thats fifteen dollars",
             "events": "on thursday we have karaoke and on tuesdays we have trivia",
             "allergins": "the only item on the menu common allergen is hummus which contain pine nuts",
             "info": "the resteraunt was founded by two brothers in two thousand and three"}

    prompt = 'what pasta dishes do you have'
    print(f'finding relevent doc for "{prompt}"')
    print(retreive_relevent(prompt, documents))
    print('----')
    prompt = 'what events do you guys do'
    print(f'finding relevent doc for "{prompt}"')
    print(retreive_relevent(prompt, documents))

    prompt = 'what events do you guys do'
    print(f'prompt for "{prompt}":\n')
    print(retreive_and_agument(prompt, documents))

if __name__ == '__main__':
    main()