import time
import random
from sentence_transformers import SentenceTransformer, util
import string
# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate dummy sentences (with some variation)
def generate_sentences(n_sentences, max_words_per_sentence=10, max_word_length=8):
    sentences = []
    for _ in range(n_sentences):
        num_words = random.randint(1, max_words_per_sentence)
        words = [
            ''.join(random.choices(string.ascii_lowercase, k=random.randint(1, max_word_length)))
            for _ in range(num_words)
        ]
        sentence = ' '.join(words)
        sentences.append(sentence)
    return sentences

# Input sentence to compare
input_sentence = "In this paper, we propose a general framework to connect planning problems with OWL ontologies"

# List of sizes to test
sizes_to_test = [10, 100, 500, 1000, 5000, 10000]

# Run tests
for size in sizes_to_test:
    sentences = generate_sentences(size)
    sentences.append("I love studying AI")
    # Timing the inference
    start = time.time()
    input_embedding = model.encode(input_sentence, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embedding, sentence_embeddings)
    most_similar_idx = cosine_scores.argmax()
    end = time.time()

    print(f"Size: {size:6} | Time taken: {end - start:.4f} sec | Most similar: {sentences[most_similar_idx]}")
