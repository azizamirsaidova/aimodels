import nltk
import numpy as np
import re
import string
import pandas as pd 
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.stats import entropy
from gensim.utils import simple_preprocess


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub('[^.,a-zA-Z0-9 \n\.]', '', text.lower())
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
    return filtered_tokens

def calculate_entropy(tokens):
    token_counts = Counter(tokens)
    probabilities = [count / len(tokens) for count in token_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy

def _range(series):
    return series.max() - series.min()

def main():
    text = "As a virtual assistant you are dedicated to sharing knowledge and opinions on social media."
    tokens = preprocess_text(text)
    entropy_res = calculate_entropy(tokens)

    df = pd.read_csv('/Users/azizamirsaidova/Downloads/output.csv')
    print(f"words: {tokens}")
    print(f"entropy: {entropy_res}")
    print(df[['entropy']].agg(['min', 'mean', 'std', _range]))

if __name__ == '__main__':
    main()