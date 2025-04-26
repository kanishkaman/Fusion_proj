import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import traceback      # For detailed error printing

# --- Global variable for the model (to load only once) ---

# Use a relatively lightweight but effective model suitable for semantic search
MODEL_NAME = 'multi-qa-MiniLM-L6-cos-v1'
embedding_model = None
faq_embeddings = None
faq_data = None

# --- Data Loading and Embedding Generation ---
def load_and_embed_faq(filepath):
    """Loads the FAQ dataset and generates embeddings for the questions."""
    global embedding_model, faq_embeddings, faq_data

    # Avoid reloading if already loaded
    if faq_data is not None and faq_embeddings is not None:
        # print("FAQ data and embeddings already loaded.") # Optional debug print
        return True

    try:
        if not os.path.exists(filepath):
             print(f"Error: FAQ file not found at {filepath}")
             faq_data = None
             faq_embeddings = None
             return False

        # Loading our Sentence-Transformer model
        if embedding_model is None:
             print(f"Loading sentence transformer model: {MODEL_NAME}...")
             # Consider specifying device='cpu' if CUDA/GPU issues persist or aren't needed
             # embedding_model = SentenceTransformer(MODEL_NAME, device='cpu')
             embedding_model = SentenceTransformer(MODEL_NAME)
             print("Sentence Transformer Model loaded.")

        # Load the FAQ data
        print(f"Loading FAQ data from: {filepath}")
        df = pd.read_csv(filepath)
        # Drop rows where essential columns might be missing
        df = df.dropna(subset=['Question Keywords', 'Answer'])
        df['Question Text'] = df['Question Keywords'].astype(str).str.lower().str.strip()
        faq_data = df       # Store globally

        # Generate embeddings for all questions in the FAQ
        print("Generating embeddings for FAQ questions...")
        questions_list = faq_data['Question Text'].tolist()
        questions_list = [str(q) for q in questions_list]

        faq_embeddings = embedding_model.encode(questions_list, convert_to_tensor=True)
        print(f"Embeddings generated for {len(questions_list)} questions.")
        return True

    #trying to catch certain errors
    except FileNotFoundError:
        print(f"Error: FAQ file not found at {filepath}")
        faq_data = None
        faq_embeddings = None
        return False
    except Exception as e:
        print(f"Error loading FAQ data or generating embeddings: {e}")
        traceback.print_exc()     # Print full traceback for debugging
        faq_data = None
        faq_embeddings = None
        return False

# --- Chatbot Logic (Sentence Similarity) ---
def get_bot_response(query, faq_filepath):
    """
    Gets a chatbot response by finding the most semantically similar
    question in the FAQ dataset.
    """
    global embedding_model, faq_embeddings, faq_data

    if faq_data is None or faq_embeddings is None:
        print("FAQ data/embeddings not loaded, attempting load...")
        if not load_and_embed_faq(faq_filepath):
             return "Sorry, my knowledge base isn't loaded correctly right now."

        if faq_data is None or faq_embeddings is None:
             print("FAQ data/embeddings failed to load.") # Debug print
             return "Sorry, my knowledge base couldn't be loaded."

    if not query:
        return "Please ask a question."

    query_lower = query.lower().strip()
    print(f"Searching for similarity with query: '{query_lower}'") # Debug print

    try:
        query_embedding = embedding_model.encode(query_lower, convert_to_tensor=True)

        # Compute cosine similarity between the query and all FAQ questions
        # Ensure embeddings are on the same device if using GPU
        # cosine_scores = util.cos_sim(query_embedding.to(faq_embeddings.device), faq_embeddings)[0]
        cosine_scores = util.cos_sim(query_embedding, faq_embeddings)[0]

        # Find index (row number in faq_data) of the highest score
        best_match_idx = torch.argmax(cosine_scores).item()
        best_score = cosine_scores[best_match_idx].item()

                ## Similarity threshold
        SIMILARITY_THRESHOLD = 0.4 # Lower this from 0.5 if too many "no match" cases aris; make it higher (e.g., 0.6) if matching unrelated questions

        print(f"Best match index: {best_match_idx}, Score: {best_score:.4f}, Threshold: {SIMILARITY_THRESHOLD}") # Debug print

        if best_score >= SIMILARITY_THRESHOLD:
            # Retrieve the answer corresponding to the best match index
            answer = faq_data.iloc[best_match_idx]['Answer']
            matched_question = faq_data.iloc[best_match_idx]['Question Text']
            print(f"Match found: '{matched_question}'")
            return answer
        else:
            print(f"No match above threshold.")
            return "Sorry, I couldn't find a close match for that question in my knowledge base. Please try rephrasing or asking about general nutrition topics."

    except Exception as e:
        print(f"Error during similarity search or response generation: {e}")
        traceback.print_exc()
        return "Sorry, I encountered an error trying to find an answer."

