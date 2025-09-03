from sentence_transformers import SentenceTransformer, util

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # small & fast, yet decent

# Your list of available sentences
sentence_list = [
    "I love machine learning.",
    "The sky is blue today.",
    "Cats are great companions.",
    "Artificial intelligence is fascinating.",
    "I'm going to the grocery store."
]

# Your input sentence
input_sentence = "I think AI is really interesting."

# Encode sentences into embeddings
input_embedding = model.encode(input_sentence, convert_to_tensor=True)
list_embeddings = model.encode(sentence_list, convert_to_tensor=True)

# Compute cosine similarities
cosine_scores = util.cos_sim(input_embedding, list_embeddings)

# Find the most similar sentence
most_similar_idx = cosine_scores.argmax()
most_similar_sentence = sentence_list[most_similar_idx]

print("Input:", input_sentence)
print("Most similar:", most_similar_sentence)
