import nltk
import re
import string
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer

# Step 1: Read the file
file_name = input("Enter Your file name : ")
with open(file_name, 'r') as f:
    data = f.read().lower()
    uploaded_words = re.findall(r'\w+', data)

print("Uploaded words:", uploaded_words)

# Step 2: Count word frequencies
def count_word_frequency(words):
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    return word_count

word_count = count_word_frequency(uploaded_words)
print("\nWord count:", word_count)

# Step 3: Calculate word probabilities
def count_probability(word_count):
    total_words = sum(word_count.values())
    return {word: count / total_words for word, count in word_count.items()}

probabilities = count_probability(word_count)

# Step 4: Build vocabulary
english_vocab = set(w.lower() for w in words.words())
vocab = english_vocab.union(set(word_count.keys()))

lemmatizer = WordNetLemmatizer()

# Step 5: Edit distance operations
def delete_letter(word):
    return [word[:i] + word[i+1:] for i in range(len(word))]

def swap_letter(word):
    return [word[:i] + word[i+1] + word[i] + word[i+2:]
            for i in range(len(word) - 1)]

def replace_letter(word):
    letters = string.ascii_lowercase
    return [word[:i] + l + word[i+1:]
            for i in range(len(word)) for l in letters]

def insert_letters(word):
    letters = string.ascii_lowercase
    return [word[:i] + l + word[i:] for i in range(len(word) + 1) for l in letters]

# Step 6: Generate possible candidates
def generate_candidate(word):
    candidate = set()
    candidate.update(delete_letter(word))
    candidate.update(swap_letter(word))
    candidate.update(replace_letter(word))
    candidate.update(insert_letters(word))
    return candidate

# Step 7: Generate candidates with two edits
def generate_candidate_level2(word):
    level1 = generate_candidate(word)
    level2 = set()
    for w in level1:
        level2.update(generate_candidate(w))
    return level2

# Step 8: Suggest best correction(s)
def get_best_correction(word, probs, vocab, max_suggestions=3):
    if word in vocab:
        candidates = [word]
    else:
        candidates = list(generate_candidate(word).intersection(vocab))

    if not candidates:
        # If no close match found at one edit distance, check level 2
        candidates = list(generate_candidate_level2(word).intersection(vocab))

    # Sort by probability (if present) or alphabetically
    sorted_candidates = sorted(
        [(w, probs.get(w, 0)) for w in candidates],
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_candidates[:max_suggestions]

# Step 9: Take user input and show suggestions
user_input = input("\nEnter a word for autocorrection: ").lower()
suggestions = get_best_correction(user_input, probabilities, vocab, max_suggestions=5)

print("\nTop suggestions:")
if suggestions:
    for suggestion in suggestions:
        print(f"→ {suggestion[0]}")
else:
    print("No suggestions found.")
