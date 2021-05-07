import torch
from transformers import BertTokenizer, BertModel
from transformers.tokenization_bert import BertTokenizer
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)
import nltk
import pandas as pd
#from bert import Ner
import matplotlib.pyplot as plt
plt.show()

read_text = pd.read_csv(r"C:\Users\ehcho\ner_bert\data\imbd_csv\imdb_train.csv", sep="\t", encoding="UTF-8",
                   names=['pn', 'review'], header=None)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

text = "The Mascot is Ladislaw Starewicz's masterpiece. It tells the tale of a stuffed toy dog who searches for an orange after he overhears the mother tell her daughter she hasn't any money. The dog gets picked with other toys to be sold off. In the truck, after the others jump off while the vehicle runs, the dog stays and waits to be picked up from store before setting off on his own. He manages to get orange after biting woman's leg as she was holding and selling the fruit. As he runs, he encounters the devil and accepts his offer to stop at nightclub. There, he meets the other toys who jumped off truck. The cat who was next to him in vehicle is especially determined to get dog's orange. I'll just stop here to mention that other wonderfully bizarre things happen in nightclub that you'll have to see for yourself. Suffice it to say, if you love Starewicz and is interested in all animation from the past, I most definitely recommend you seek this one out!"

marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

#tokenized_text = list(tokenizer.vocab.keys())[5000:5020]
# Print out the tokens.
print (tokenized_text)


#marked_text = "[CLS] " + text + " [SEP]"

# Split the sentence into tokens.
#tokenized_text = tokenizer.tokenize(marked_text)

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))
    
# Mark each of the 22 tokens as belonging to sentence "1".
segments_ids = [1] * len(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
print (segments_ids)


# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-large-uncased',
                                  output_hidden_states = True # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model_eval = model.eval()

print(model_eval)


#BERT Base: 12 layers (transformer blocks), 12 attention heads, and 110 million parameters
#BERT Large: 24 layers (transformer blocks), 16 attention heads and, 340 million parameters


# Run the text through BERT, and collect all of the hidden states produced
# from all 12 layers.
with torch.no_grad():

    outputs = model(tokens_tensor, segments_tensors)

    # Evaluating the model will return a different number of objects based on
    # how it's  configured in the `from_pretrained` call earlier. In this case,
    # becase we set `output_hidden_states = True`, the third item will be the
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]


print ("Number of layers:", len(hidden_states), "  (initial embeddings + 24 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))


# For the 5th token in our sentence, select its feature values from layer 5.
token_i = 5
layer_i = 5
vec = hidden_states[layer_i][batch_i][token_i]

# Plot the values as a histogram to show their distribution.
plt.figure(figsize=(10,10))
plt.hist(vec, bins=200)
plt.show()

# `hidden_states` is a Python list.
print('      Type of hidden_states: ', type(hidden_states))

# Each layer in the list is a torch tensor.
print('Tensor shape for each layer: ', hidden_states[0].size())