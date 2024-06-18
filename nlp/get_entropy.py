import nltk
import numpy as np
import re
import string
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

def main():
    text = "As a virtual assistant you are dedicated to sharing knowledge and opinions on social media."
    tokens = preprocess_text(text)
    entropy_res = calculate_entropy(tokens)

    print(f"words: {tokens}")
    print(f"entropy: {entropy_res}")

if __name__ == '__main__':
    main()