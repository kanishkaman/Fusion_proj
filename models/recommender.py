import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback # For better error printing

# --- Constants ---
# Define calorie bins
CALORIE_BINS = {
    "Any": None, # Explicitly add 'Any' option mapping
    "Low (<400)": (0, 400),
    "Medium (400-700)": (400, 700),
    "High (>700)": (700, float('inf'))
}

# *** NEW: Define protein bins (grams per serving) ***
PROTEIN_BINS = {
    "Any": None, # Explicitly add 'Any' option mapping
    "Low (<15g)": (0, 15),
    "Medium (15-30g)": (15, 30),
    "High (>30g)": (30, float('inf'))
}

# Define potential tag columns (used in fallback loading)
TAG_COLUMNS = [
    'vegetarian', 'vegan', 'glutenFree', 'dairyFree', 'peanutFree',
    'soyFree', 'treeNutFree', 'kosher', 'healthy', 'lowCal', 'lowFat',
    'lowSodium', 'dessert', 'drink'
]


# --- Data Loading ---
def load_cleaned_recipe_data(filepath):
    """Loads the CLEANED recipe dataset and validates essential columns."""
    try:
        # Define essential columns needed for filtering and display
        required_cols = ['title', 'calories', 'protein', 'fat', 'sodium']
        # Define columns to attempt loading (including optional rating and tags)
        cols_to_load = required_cols + ['rating'] + TAG_COLUMNS

        # Attempt to load only necessary columns first
        try:
            # Check which columns actually exist in the CSV header without loading full data
            with open(filepath, 'r', encoding='utf-8') as f: # Specify encoding
                header = f.readline().strip().split(',')
            loadable_cols = [col for col in cols_to_load if col in header]
            df = pd.read_csv(filepath, usecols=loadable_cols, encoding='utf-8')
        except Exception as e:
            print(f"Warning: Could not selectively load columns, attempting full load. Error: {e}")
            df = pd.read_csv(filepath, encoding='utf-8')


        # Validate essential columns are present after loading
        if not all(col in df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df.columns]
             raise ValueError(f"Cleaned data missing essential columns: {missing}")

        # Convert numeric columns, coercing errors
        for col in ['calories', 'protein', 'fat', 'sodium', 'rating']:
             if col in df.columns:
                 # Ensure column exists before conversion
                 df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where essential numeric conversions failed (resulted in NaN)
        # Check existence again in case a column was entirely non-numeric
        valid_numeric_cols = [col for col in ['calories', 'protein', 'fat', 'sodium'] if col in df.columns]
        df = df.dropna(subset=valid_numeric_cols)

        # Ensure essential numeric columns have valid (>0) values after cleaning
        if 'calories' in df.columns: df = df[df['calories'] > 0]
        if 'protein' in df.columns: df = df[df['protein'] >= 0] # Allow 0 protein

        print(f"Successfully loaded and validated cleaned data: {len(df)} rows.")
        return df
    except FileNotFoundError:
        print(f"Error: Cleaned recipe file not found at {filepath}")
        return None
    except ValueError as ve: # Catch specific validation error
        print(f"Data validation error: {ve}")
        return None
    except Exception as e: # Catch other potential errors during loading/parsing
        print(f"Error loading cleaned recipe data: {e}")
        traceback.print_exc()
        return None

# --- Feature Engineering & Model Prep ---
def prepare_recommender(df):
    """Prepares TF-IDF vectorizer and matrix from recipe titles."""
    if df is None or df.empty:
        print("DataFrame is empty, cannot prepare recommender.")
        return None, None
    # Ensure 'title' column exists and is string type
    if 'title' not in df.columns:
         print("Error: 'title' column missing for TF-IDF preparation.")
         return None, None
    df['title_processed'] = df['title'].fillna('').astype(str).str.lower()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['title_processed'])
        print("TF-IDF matrix prepared successfully.")
        return tfidf_vectorizer, tfidf_matrix
    except Exception as e:
        print(f"Error creating TF-IDF matrix: {e}")
        traceback.print_exc()
        return None, None


# --- Recommendation Logic (UPDATED with Protein Filter) ---
def get_recommendations(prefs, df, vectorizer, tfidf_matrix, max_results=50):
    """
    Generates recipe recommendations based on user preferences (including protein).
    Returns up to max_results.
    """
    if df is None or df.empty or vectorizer is None or tfidf_matrix is None:
        print("Recommender called with invalid data or model components.")
        return pd.DataFrame()

    # Start with a copy of the original dataframe
    filtered_df = df.copy()
    print(f"Starting recommendations with {len(filtered_df)} recipes.")
    print(f"Preferences received: {prefs}") # Debug print

    # --- 1. Filter by Hard Constraints ---

    # Calorie Filtering
    selected_calorie_bin = prefs.get("calories") # e.g., "Low (<400)"
    if selected_calorie_bin and selected_calorie_bin != "Any": # Check if not None and not "Any"
        if selected_calorie_bin in CALORIE_BINS:
            min_cal, max_cal = CALORIE_BINS[selected_calorie_bin]
            if 'calories' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['calories']):
                filtered_df = filtered_df[(filtered_df['calories'] >= min_cal) & (filtered_df['calories'] < max_cal)]
                print(f"After calorie filter ('{selected_calorie_bin}'): {len(filtered_df)} recipes.")
            else:
                 print("Warning: 'calories' column not found or not numeric, skipping calorie filter.")
        else:
             print(f"Warning: Invalid calorie bin selected: {selected_calorie_bin}")


    # *** NEW: Protein Filtering ***
    selected_protein_bin = prefs.get("protein") # e.g., "High (>30g)"
    if selected_protein_bin and selected_protein_bin != "Any": # Check if not None and not "Any"
        if selected_protein_bin in PROTEIN_BINS:
            min_prot, max_prot = PROTEIN_BINS[selected_protein_bin]
            # Ensure 'protein' column is numeric before filtering
            if 'protein' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['protein']):
                filtered_df = filtered_df[(filtered_df['protein'] >= min_prot) & (filtered_df['protein'] < max_prot)]
                print(f"After protein filter ('{selected_protein_bin}'): {len(filtered_df)} recipes.")
            else:
                 print("Warning: 'protein' column not found or not numeric, skipping protein filter.")
        else:
             print(f"Warning: Invalid protein bin selected: {selected_protein_bin}")


    # Dietary Tag Filtering
    selected_tags = prefs.get("dietary_tags", [])
    if selected_tags:
        for tag in selected_tags:
            if tag in filtered_df.columns:
                 # Assume tags are 0/1 after cleaning
                 if pd.api.types.is_numeric_dtype(filtered_df[tag]):
                     # Ensure filtering works correctly for boolean/int types
                     try:
                         filtered_df = filtered_df[filtered_df[tag].astype(bool)] # Filter where tag is True (or 1)
                         print(f"After applying tag '{tag}': {len(filtered_df)} recipes.")
                     except Exception as tag_filter_e:
                          print(f"Warning: Could not apply filter for tag '{tag}': {tag_filter_e}")
                 else:
                      print(f"Warning: Tag column '{tag}' is not numeric, skipping filter.")
            # else: # Reduce noise by not warning for every possible tag not present
            #     pass


    if filtered_df.empty:
        print("No recipes found matching the hard constraints.")
        return pd.DataFrame()

    # --- 2. Filter/Rank by Keyword Preference (Soft Constraint using TF-IDF) ---
    keyword_pref = prefs.get("keywords", "")
    recommendations = pd.DataFrame() # Initialize

    # Check if TF-IDF components are valid before proceeding
    if vectorizer is None or tfidf_matrix is None:
        print("Warning: TF-IDF model not available. Falling back to rating sort.")
        keyword_pref = "" # Skip keyword search if model invalid

    if keyword_pref:
        print(f"Applying keyword search: '{keyword_pref}'")
        try:
            # Ensure 'title_processed' exists before transforming
            if 'title_processed' not in filtered_df.columns:
                 # Attempt to create it on the fly if missing (should have been done in prepare_recommender)
                 if 'title' in filtered_df.columns:
                      filtered_df['title_processed'] = filtered_df['title'].fillna('').astype(str).str.lower()
                 else:
                      raise ValueError("'title' column missing to create 'title_processed'.")

            keyword_vec = vectorizer.transform([keyword_pref.lower()])
            original_indices = filtered_df.index

            # Map filtered indices back to original TF-IDF matrix positions
            try:
                # Get positional indices based on the original index values present in filtered_df
                # This requires the base df used for TF-IDF prep had a standard index (0, 1, ...)
                # Use get_indexer for robustness if original index is not sequential
                positional_indices = df.index.get_indexer(original_indices)
                # Filter out -1 which indicates index not found (shouldn't happen if filtered_df is subset of df)
                positional_indices = positional_indices[positional_indices != -1]

                if len(positional_indices) == 0:
                     print("Warning: No matching indices found for similarity calculation.")
                     recommendations = pd.DataFrame() # No matches
                else:
                    subset_tfidf_matrix = tfidf_matrix[positional_indices]
                    cosine_similarities = cosine_similarity(keyword_vec, subset_tfidf_matrix).flatten()
                    # Get indices within the *filtered* subset, sorted by similarity
                    similar_indices_in_subset = cosine_similarities.argsort()[::-1]
                    # Get the final recommended indices from the filtered_df based on similarity sort order
                    top_filtered_indices = filtered_df.iloc[similar_indices_in_subset].index

                    # Select up to max_results based on similarity
                    recommendations = df.loc[top_filtered_indices[:max_results]]
                    print(f"Found {len(recommendations)} recipes matching keywords, sorted by similarity.")

            except (KeyError, IndexError, ValueError) as e: # Catch potential indexing errors
                 print(f"Index mapping/subset error during similarity calculation: {e}. Falling back.")
                 traceback.print_exc()
                 # Fallback: Filter by keyword presence and sort by rating
                 keyword_pattern = '|'.join([f'\\b{word}\\b' for word in keyword_pref.lower().split()])
                 keyword_matches = filtered_df[filtered_df['title_processed'].str.contains(keyword_pattern, case=False, na=False)]
                 if not keyword_matches.empty and 'rating' in keyword_matches.columns:
                     recommendations = keyword_matches.sort_values(by='rating', ascending=False).head(max_results)
                 else:
                     recommendations = keyword_matches.head(max_results)
                 print(f"Fallback keyword search yielded {len(recommendations)} recipes.")

        except Exception as e: # Catch other errors during keyword processing/similarity
            print(f"Error processing keywords or calculating similarity: {e}")
            traceback.print_exc()
            # Fall back to filtering without keywords, sort by rating
            if not filtered_df.empty and 'rating' in filtered_df.columns:
                 recommendations = filtered_df.sort_values(by='rating', ascending=False).head(max_results)
            else:
                 recommendations = filtered_df.head(max_results)

    else:
        # If no keywords, just return top N from filtered list, sorted by rating if available
        print("No keywords provided or TF-IDF unavailable. Sorting filtered results by rating.")
        if not filtered_df.empty and 'rating' in filtered_df.columns:
            recommendations = filtered_df.sort_values(by='rating', ascending=False).head(max_results)
        else:
            # Fallback if no rating column exists or df is empty
            recommendations = filtered_df.head(max_results) # Simple head or sample

    # Ensure we return a DataFrame even if empty
    if recommendations.empty:
        print("No recommendations generated.")
        return pd.DataFrame()

    # --- Return relevant columns ---
    # Define columns to return
    return_cols = ['title', 'rating', 'calories', 'protein', 'fat', 'sodium']
    # Keep only columns that actually exist in the recommendations DataFrame
    existing_return_cols = [col for col in return_cols if col in recommendations.columns]

    return recommendations[existing_return_cols]
