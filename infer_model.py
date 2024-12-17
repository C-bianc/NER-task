#!/usr/bin/env python
# ~* coding: utf-8 *~
#===============================================================================
#
#           FILE: infer_model.py 
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 25-05-2024 
#
#===============================================================================
#    DESCRIPTION:  infers the trained model and prints the predicted labels for each word
#                  uses the trained RobertaForTokenClassification
#    
#   DEPENDENCIES:  pip install -r requirements.txt
#
#          USAGE: python infer_model.py
#
#===============================================================================
print("Loading...")
import torch
import torch.nn.functional as F
import warnings
from transformers import RobertaForTokenClassification, RobertaConfig, RobertaTokenizerFast
import random
import string
from collections import Counter
import sys
import os
import csv

global n_queries
n_queries = 0

tagset_id2label = {0: 'O', 1: 'B-ORG', 2: 'I-ORG', 3: 'B-MISC', 4: 'I-MISC', 5: 'B-PER', 6: 'I-PER', 7: 'B-LOC', 8: 'I-LOC'}

warnings.filterwarnings("ignore", category=FutureWarning)

# load our trained model
config = RobertaConfig.from_pretrained('roberta-base', num_labels=len(tagset_id2label))
model = RobertaForTokenClassification(config)
model.load_state_dict(torch.load("./model/TrainedRoberta4TokenClass.pth"))
model.eval()

# load the saved tokenizer 
tokenizer = RobertaTokenizerFast.from_pretrained("./model/simple_tokenizer")

def align_tokens_with_prediction(tokens, labels, original_pos, scores):
    """ 
        long but simple algorithm for glueing word-pieces to their corresponding entity label 
        we take tokens which are word-pieces and their predicted labels
        we count how many times an index is repeated in original_pos and use this as indx to glue tokens
    """

    tokens = tokens[1:-1]
    tokens = [token.replace("Ġ","") for token in tokens]

    labels = labels[1:-1]
    original_pos = original_pos[1:-1]
    word_pieces_ids = Counter(original_pos)

    final_tokens = []
    final_labels = []
    final_scores = []
  
    i = 0
    n_iter = 0
    while n_iter < len(word_pieces_ids):

        jumps = word_pieces_ids[n_iter] # for this current token, how many pieces, thus how many to skip
        original_word = ''.join(tokens[i: i + jumps])
        label = labels[i]
        score = scores[i]

        final_tokens.append(original_word)
        final_labels.append(label)
        final_scores.append(score)

        i += jumps
        n_iter += 1
                 
    return final_tokens, final_labels



def infer_model(query):
    # model input
    inputs = tokenizer(query, return_tensors="pt")
    original_positions = inputs.word_ids()

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_labels = outputs.logits.argmax(dim=-1).squeeze().tolist()

        probabilities = F.softmax(outputs.logits, dim=-1)
        scores = torch.max(probabilities, dim=-1).values.squeeze().tolist()

        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
        predicted_labels = [tagset_id2label[label_id] for label_id in predicted_labels]
 
    return tokens, predicted_labels, original_positions, scores


def get_user_input():
    print("1. Get named-entities tags from a sentence in English")
    print("2. Quit")
    print("3. Get a Bob Ross quote ¯\_(ツ)_/¯")
    print("Type 'quit' or 'exit' to stop the program at any time\n")

    choice = input("Choose your fighter: ")
    return choice

def display_results(tokens, predictions, scores):

    max_len = len(max(tokens, key=len)) + 10

    print(f'\n     Tokens {" " * (max_len // 2)} \tPredicted label   \t Confidence score')

    for token, label, score in zip(tokens, predictions, scores):
        print(f'{token:{max_len}} \t    {label:{max_len}}  \t     {score} %') 

    print("\nResults stored in 'query_results' dir!\n")

def write_results(tokens, predictions, scores):
    output_dir = "./query_results/"
    os.makedirs(output_dir, exist_ok=True)

    suffix = ''.join(random.choices(string.ascii_letters, k=4))

    with open(os.path.join(output_dir, f'query_{n_queries}_{suffix}_results.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Token', 'Prediction', 'Confidence_score'])

        # write to file
        for token, prediction, score in zip(tokens, predictions, scores):
            writer.writerow([token, prediction, score/100])

bob_ross_quotes = [
    "We don't make mistakes, just happy little accidents.",
    "There are no limits here, you start out by believing here.",
    "Talent is a pursued interest. Anything that you're willing to practice, you can do.",
    "Let's get crazy.",
    "You need the dark in order to show the light.",
    "We want happy paintings. Happy paintings. If you want sad things, watch the news.",
    "In painting, you have unlimited power. You have the ability to move mountains. You can bend rivers."
]


# code to run when program is launched
if __name__ == "__main__":
    print("Welcome to a simple NER Tagging Tool! Created and model (RobertaForTokenClassification) trained by Bianca.")
    print("The results will be stored in 'query_results' dir\n")
    print("Choose among the following options\n")

    while True:
        choice = get_user_input().strip()
        if choice == '1':
            while True:
                query = input("Enter a sentence :\n")

                if query.lower() in ['quit', 'exit']:
                    print("Exiting the program.")
                    exit()
                # infer model
                tokens, predictions, original_pos, scores = infer_model(query)
                scores = [round(score * 100, 2) for score in scores]

                # some fancy alignment
                tokens, predictions = align_tokens_with_prediction(tokens, predictions, original_pos, scores)
                display_results(tokens, predictions, scores)
                write_results(tokens, predictions, scores)
                n_queries += 1

        elif choice == '2' or choice == 'quit' or choice == 'exit':
            print("Exiting the program.")
            break

        elif choice == '3':
            print("__________________________________________")
            print("\nBob Ross says: ", random.choice(bob_ross_quotes), "\n")
            print("__________________________________________")
            next

        else:
            print("__________________________________________")
            print("\nInvalid choice. Please enter 1, 2 or 3.\n")
