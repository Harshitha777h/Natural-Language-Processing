import os
import re
import random
from collections import defaultdict, Counter

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

def build_ngram_model(tokens, n=5):
    model = defaultdict(Counter)

    for i in range(len(tokens) - n + 1):
        context = tuple(tokens[i:i + n - 1])   # previous 4 words
        next_word = tokens[i + n - 1]           # next word
        model[context][next_word] += 1

    return model

def predict_next_word(model, context):
    if context not in model:
        return None

    words, counts = zip(*model[context].items())
    return random.choices(words, weights=counts)[0]

def generate_text(model, seed_text, num_words=40):
    seed_tokens = preprocess_text(seed_text)

    if len(seed_tokens) < 4:
        return "Error: Seed text must contain at least 4 words."

    generated = seed_tokens[:]

    for _ in range(num_words):
        context = tuple(generated[-4:])
        next_word = predict_next_word(model, context)

        if next_word is None:
            break

        generated.append(next_word)

    return " ".join(generated)

def load_corpus(folder_path):
    all_text = ""

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, encoding="utf-8") as f:
                all_text += f.read() + " "

    return all_text

if __name__ == "__main__":

    print("Loading Lovecraft corpus...")
    corpus_text = load_corpus("lovecraft_corpus")

    print("Preprocessing text...")
    tokens = preprocess_text(corpus_text)

    print("Building 5-gram language model...")
    ngram_model = build_ngram_model(tokens, n=5)

    print("\nModel ready.")
    print("Enter seed text (minimum 4 words). Type 'exit' to quit.\n")

    while True:
        seed = input("> ")

        if seed.lower() == "exit":
            print("Exiting program.")
            break

        output = generate_text(ngram_model, seed)
        print("\nGenerated Text:")
        print(output)
        print("-" * 60)

