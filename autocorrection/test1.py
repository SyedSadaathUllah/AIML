import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words

# --- Step 1: Load your text file ---
filename = input("Enter the path of your file: ")
with open(filename, 'r') as f:
    data = f.read().lower()
    uploaded_words = re.findall(r'\w+', data)

print("File content loaded successfully!\n")
print("Words from file:", uploaded_words)

# --- Step 2: Count word frequency from your file ---
def count_word_frequency(words):
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    return word_count

word_count = count_word_frequency(uploaded_words)

# --- Step 3: Calculate probabilities for your uploaded words ---
def count_probability(word_count):
    total_words = sum(word_count.values())
    return {word: count / total_words for word, count in word_count.items()}

probabilities = count_probability(word_count)

# --- Step 4: Build a large vocabulary using NLTK's corpus + your file words ---
english_vocab = set(w.lower() for w in words.words())
vocab = english_vocab.union(set(word_count.keys()))

lemmatizer = WordNetLemmatizer()

# --- Step 5: Define word-editing functions for generating candidates ---
def delete_letter(word):
    return [word[:i] + word[i+1:] for i in range(len(word))]

def swap_letter(word):
    return [word[:i] + word[i+1] + word[i] + word[i+2:] for i in range(len(word)-1)]

def replace_letter(word):
    letters = string.ascii_lowercase
    return [word[:i] + l + word[i+1:] for i in range(len(word)) for l in letters]

def insert_letter(word):
    letters = string.ascii_lowercase
    return [word[:i] + l + word[i:] for i in range(len(word)+1) for l in letters]

# --- Step 6: Generate candidate corrections ---
def generate_candidates(word):
    candidates = set()
    candidates.update(delete_letter(word))
    candidates.update(swap_letter(word))
    candidates.update(replace_letter(word))
    candidates.update(insert_letter(word))
    return candidates

def generate_candidates_level2(word):
    level1 = generate_candidates(word)
    level2 = set()
    for w in level1:
        level2.update(generate_candidates(w))
    return level2

# --- Step 7: Suggest best corrections ---
def get_best_correction(word, probs, vocab, max_suggestions=3):
    if word in vocab:
        candidates = [word]
    else:
        candidates = list(generate_candidates(word).intersection(vocab))

    if not candidates:
        # If no close match found at one edit distance, check level 2
        candidates = list(generate_candidates_level2(word).intersection(vocab))

    # Sort by probability (if present in your file data) or alphabetically
    return sorted([(w, probs.get(w, 0)) for w in candidates],
                  key=lambda x: x[1],
                  reverse=True)[:max_suggestions]

# --- Step 8: Take user input and show suggestions ---
user_input = input("\nEnter a word for autocorrection: ").lower()
suggestions = get_best_correction(user_input, probabilities, vocab, max_suggestions=5)

print("\nTop suggestions:")
if suggestions:
    for suggestion in suggestions:
        print(f"→ {suggestion[0]}")
else:
    print("No suggestions found.")