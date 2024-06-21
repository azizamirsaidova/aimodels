# Install the required libraries
# !pip install transformers

from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(user_input):
    # Tokenize the input
    input_ids = tokenizer.encode(user_input, truncation=True, padding=True)

def classify_intent(input_ids):
    
    # Predict the intent
    logits = model(torch.tensor(input_ids).unsqueeze(0))[0]
    intent_id = logits.argmax().item()
    # Map the intent ID to a human-readable label
    intent_label = ['Positive', 'Negative'][intent_id]
    return intent_label

def classify_sentiment(input_ids):
    outputs = model(**input_ids)
    predicted_class = torch.argmax(outputs.logits)
    return predicted_class


user_input = "I love this product!"
input_ids = tokenize(user_input)
# print(classify_intent(input_ids)) 

print("Predicted Sentiment Class:", classify_sentiment(input_ids))