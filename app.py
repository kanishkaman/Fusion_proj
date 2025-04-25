# Import necessary libraries
import streamlit as st
import pandas as pd
# Import custom modules for recommender and chatbot
from models import recommender, chatbot
import os # For path operations
import traceback # For detailed error printing

# --- Page Configuration ---
st.set_page_config(
    page_title="Mindful Meal Planner", # Kept original title, can be changed if needed
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Highlighting ---
# Applying styles using st.markdown
st.markdown("""
<style>
    /* Style for the chatbot container */
    .chatbot-container {
        background-color: rgba(230, 240, 255, 0.7); /* Light blueish, semi-transparent */
        border-radius: 10px;
        padding: 15px 20px 20px 20px; /* Added more bottom padding */
        margin-bottom: 20px;
        border: 1px solid rgba(0, 123, 255, 0.2);
        overflow: hidden; /* Prevents content spillover affecting layout */
    }
    /* Style for the creators container inside the expander */
    .creators-container {
        background-color: rgba(255, 240, 230, 0.7); /* Light orangish, semi-transparent */
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(255, 123, 0, 0.2);
    }
    /* Style for chat messages within the container */
    .chatbot-container [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 8px;
        margin-bottom: 10px;
    }
    /* Style for example buttons */
    .stButton>button {
        font-size: 0.9rem; /* Slightly smaller font for buttons */
        padding: 0.3rem 0.6rem; /* Adjust padding */
    }
    /* Center the placeholder image */
    .recipe-placeholder img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        opacity: 0.7; /* Make it slightly transparent */
    }
</style>
""", unsafe_allow_html=True)

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RECIPE_FILE = os.path.join(DATA_DIR, "recipes_cleaned.csv")
FAQ_FILE = os.path.join(DATA_DIR, "nutrition_faq.csv") # Needed for Sentence Transformer

# --- Constants ---
RECIPES_INCREMENT = 5 # How many recipes to show per click
INITIAL_RECIPES = 5 # Initial number of recipes to show
MAX_RECIPES_TO_FETCH = 40 # Max recipes to fetch from recommender initially
PLACEHOLDER_IMAGE_URL = "https://mir-s3-cdn-cf.behance.net/projects/404/7045567.546fd07bc8521.jpg"


# --- Load Data and Prepare Models (Cached) ---

@st.cache_data
def load_recipes():
    """Loads the cleaned recipe data."""
    if not os.path.exists(RECIPE_FILE):
        return pd.DataFrame()
    try:
        df = recommender.load_cleaned_recipe_data(RECIPE_FILE)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading recipe data: {e}")
        return pd.DataFrame()

@st.cache_resource
def prepare_recipe_recommender(_df):
    """Prepares the TF-IDF vectorizer and matrix."""
    if _df.empty: return None, None
    try:
        return recommender.prepare_recommender(_df)
    except Exception as e:
        st.error(f"Error preparing recommender: {e}")
        return None, None

@st.cache_resource
def load_faq_and_model(filepath):
    """Loads FAQ data, Sentence Transformer model, and generates embeddings."""
    print("Attempting to load FAQ data and Sentence Transformer model...")
    success = chatbot.load_and_embed_faq(filepath)
    if not success: print("Failed to load FAQ/Model.")
    return success

# --- Load data and prepare models ---
recipe_df = load_recipes()
vectorizer, tfidf_matrix = prepare_recipe_recommender(recipe_df)
faq_load_successful = load_faq_and_model(FAQ_FILE)

# --- Initialize Session State ---
# Using specific keys for clarity
if 'recipe_num_shown' not in st.session_state:
    st.session_state.recipe_num_shown = INITIAL_RECIPES
if 'recipe_recommendations' not in st.session_state:
    st.session_state.recipe_recommendations = pd.DataFrame()
if 'chat_latest_query' not in st.session_state:
    st.session_state.chat_latest_query = None
if 'chat_latest_response' not in st.session_state:
    st.session_state.chat_latest_response = None
# Initialize last prefs with default 'Any' values matching the selectbox options
if 'recipe_last_prefs' not in st.session_state:
    st.session_state.recipe_last_prefs = {
        "calories": "Any",
        "protein": "Any",
        "dietary_tags": [],
        "keywords": ""
    }
# Flag to track if a search has been initiated
if 'search_initiated' not in st.session_state:
    st.session_state.search_initiated = False
if 'chat_button_clicked' not in st.session_state:
     st.session_state.chat_button_clicked = False


# --- Helper Function for Chatbot Query ---
def handle_chat_query(user_prompt):
    """Processes user query, gets response, updates session state."""
    st.session_state.chat_latest_query = user_prompt
    st.session_state.chat_latest_response = None # Clear previous response

    if faq_load_successful:
        with st.spinner("Searching knowledge base..."):
            try:
                response = chatbot.get_bot_response(user_prompt, FAQ_FILE)
                st.session_state.chat_latest_response = response
            except Exception as e:
                 st.session_state.chat_latest_response = f"Sorry, an error occurred finding an answer."
                 print(f"Error getting bot response: {e}")
                 traceback.print_exc()
    else:
        st.session_state.chat_latest_response = "Chatbot knowledge base is not available."

    # Rerun is essential here to update the chat display
    st.rerun()

# --- Streamlit UI ---
# Applying user's title change
st.title("🥗 Meal Planner & Nutrition Assistant")
st.markdown("Get recipe recommendations and nutrition advice!")

# --- Sidebar for Preferences ---
with st.sidebar: # Use 'with' for cleaner sidebar grouping
    st.header("Aaj Kya Khaoge?")

    # Initialize prefs dictionary for this run
    current_prefs = {}

    if recipe_df.empty:
        st.error(f"Recipe data ({os.path.basename(RECIPE_FILE)}) not found or empty. Recommendations unavailable.")
        # Disable recipe button if data missing
        find_button_pressed = st.button("Find Recipes", use_container_width=True, disabled=True, key="btn_find_recipes_disabled")
        # Keep last known prefs if data missing
        current_prefs = st.session_state.recipe_last_prefs

    else:
        # Calorie Preference
        calorie_options = list(recommender.CALORIE_BINS.keys())
        # Ensure last preference exists in options before getting index
        last_cal = st.session_state.recipe_last_prefs.get("calories", "Any")
        cal_index = calorie_options.index(last_cal) if last_cal in calorie_options else 0
        selected_calorie = st.selectbox(
            "Approximate Calories per Serving:",
            options=calorie_options,
            index=cal_index,
            key="sb_calories"
        )
        current_prefs["calories"] = selected_calorie

        # Protein Preference (Retained from previous version)
        if hasattr(recommender, 'PROTEIN_BINS'):
            protein_options = list(recommender.PROTEIN_BINS.keys())
            # Ensure last preference exists in options before getting index
            last_prot = st.session_state.recipe_last_prefs.get("protein", "Any")
            prot_index = protein_options.index(last_prot) if last_prot in protein_options else 0
            selected_protein = st.selectbox(
                "Approximate Protein per Serving (g):",
                options=protein_options,
                index=prot_index,
                key="sb_protein"
            )
            current_prefs["protein"] = selected_protein
        else:
            # Don't show warning if bins just aren't defined
            # st.warning("Protein bins not defined in recommender.")
            current_prefs["protein"] = "Any" # Default to Any if bins missing


        # Dietary Tags
        available_tags = []
        potential_tags = recommender.TAG_COLUMNS # Use tags from recommender module
        if not recipe_df.empty:
            available_tags = [tag for tag in potential_tags if tag in recipe_df.columns]
        if available_tags:
            selected_dietary_tags = st.multiselect(
                "Dietary Needs / Tags:",
                options=available_tags,
                default=st.session_state.recipe_last_prefs.get("dietary_tags", []),
                key="ms_tags"
            )
            current_prefs["dietary_tags"] = selected_dietary_tags
        else:
            # st.info("No specific dietary tag columns found.") # Reduce noise
            current_prefs["dietary_tags"] = []

        # Keyword Input
        user_keywords = st.text_input(
            "Keywords (e.g., 'chicken pasta'):",
            value=st.session_state.recipe_last_prefs.get("keywords", ""),
            key="ti_keywords"
        )
        current_prefs["keywords"] = user_keywords

        # Find Recipes Button
        find_button_pressed = st.button("Find Recipes", use_container_width=True, key="btn_find_recipes")

        # --- Logic for getting recommendations ---
        # Check if preferences have changed since last time
        prefs_changed = (
            current_prefs["calories"] != st.session_state.recipe_last_prefs.get("calories", "Any") or
            current_prefs["protein"] != st.session_state.recipe_last_prefs.get("protein", "Any") or
            set(current_prefs["dietary_tags"]) != set(st.session_state.recipe_last_prefs.get("dietary_tags", [])) or # Compare sets for multiselect
            current_prefs["keywords"] != st.session_state.recipe_last_prefs.get("keywords", "")
        )


        # Trigger on button press OR if preferences change (and recommender is ready)
        # if vectorizer is not None and (find_button_pressed or prefs_changed):
        if vectorizer is not None and find_button_pressed:
            # Only fetch if button pressed or if prefs actually changed
            if find_button_pressed or prefs_changed:
                st.session_state.search_initiated = True # Mark that a search has happened
                print(f"Prefs changed: {prefs_changed}, Button pressed: {find_button_pressed}")
                print(f"Current Prefs: {current_prefs}")
                print(f"Last Prefs: {st.session_state.recipe_last_prefs}")
                st.session_state.recipe_last_prefs = current_prefs.copy() # Store current prefs
                with st.spinner("Finding recipes..."):
                    try:
                        # Fetch a larger batch initially
                        all_recs = recommender.get_recommendations(
                            current_prefs, recipe_df, vectorizer, tfidf_matrix, max_results=MAX_RECIPES_TO_FETCH
                        )
                        st.session_state.recipe_recommendations = all_recs
                        st.session_state.recipe_num_shown = INITIAL_RECIPES # Reset display count
                        print(f"Fetched {len(all_recs)} total recommendations.")
                    except Exception as e:
                        st.error(f"Error getting recommendations: {e}")
                        traceback.print_exc()
                        st.session_state.recipe_recommendations = pd.DataFrame()
                # Rerun necessary if recommendations were updated due to preference change without button press
                if prefs_changed and not find_button_pressed:
                    print("Prefs changed, rerunning...")
                    st.rerun()


# --- Main Area Layout ---
col1, col2 = st.columns([2, 1])

# --- Recipe Recommendations Column ---
with col1:
    st.header("🍽️ Recipe Recommendations")
    # Display recipes only if recommender is ready and data exists
    if not recipe_df.empty and vectorizer is not None:
        recommendations_to_display = st.session_state.recipe_recommendations

        # --- Placeholder Logic ---
        # Show placeholder only if NO search has been initiated yet
        if not st.session_state.search_initiated:
            st.markdown('<div class="recipe-placeholder">', unsafe_allow_html=True) # Apply centering class
            st.image(PLACEHOLDER_IMAGE_URL, caption="Select your preferences in the sidebar, and then click on 'Find Recipes!'")
            st.markdown('</div>', unsafe_allow_html=True)
        # --- End Placeholder Logic ---

        # Display recommendations if a search HAS been initiated and results exist
        elif st.session_state.search_initiated and not recommendations_to_display.empty:
            num_to_show = st.session_state.recipe_num_shown
            st.success(f"Showing {min(num_to_show, len(recommendations_to_display))} of {len(recommendations_to_display)} recommendations:")

            for index, row in recommendations_to_display.head(num_to_show).iterrows():
                 with st.container(key=f"recipe_{index}"):
                    st.subheader(f"• {row['title']}")
                    rating_str = f"⭐ {row['rating']:.1f}/5 | " if 'rating' in row and pd.notna(row['rating']) else ""
                    try:
                        # Applying user's caption changes
                        cal_str = f"Energy ~{int(row['calories'])} kcal"
                        prot_str = f"Proteins: {row['protein']:.1f}g" if 'protein' in row and pd.notna(row['protein']) else "Proteins: N/A"
                        fat_str = f"Fats: {row['fat']:.1f}g" if 'fat' in row and pd.notna(row['fat']) else "Fats: N/A"
                        sod_str = f"Na: {row['sodium']:.0f}mg" if 'sodium' in row and pd.notna(row['sodium']) else "Na: N/A"
                        st.caption(f"{rating_str}{cal_str} | {prot_str} | {fat_str} | {sod_str}")
                    except (ValueError, TypeError, KeyError):
                        st.caption("Nutritional info unavailable.")
                    st.markdown("---")

            if num_to_show < len(recommendations_to_display):
                if st.button(f"Show {RECIPES_INCREMENT} More", key="show_more_recs_btn"):
                    st.session_state.recipe_num_shown += RECIPES_INCREMENT
                    st.rerun()
        # Message if search was initiated but no results found
        elif st.session_state.search_initiated and recommendations_to_display.empty:
             st.warning("No recipes found matching all criteria.")
        # Fallback message if search not initiated but placeholder logic somehow skipped (shouldn't happen)
        elif not st.session_state.search_initiated:
              st.info("Adjust preferences and click 'Find Recipes'.")

    else:
        # Message if data/recommender failed
        st.info("Recipe data not loaded or recommender failed. Recommendations unavailable.")


# --- Chatbot Column ---
with col2:
    # Apply the container class using markdown BEFORE elements - Restored
    # st.markdown('<div class="chatbot-container">', unsafe_allow_html=True)

    # Applying user's header and description changes
    st.header("🤖 NutriBot")
    st.markdown("Your assistant here! Ask about nutrition facts, substitutes, etc.")

    # Example Questions Buttons
    st.markdown("Try asking:")
    cols_ex = st.columns(3)
    # Using user's button text and mapping to original prompts
    example_prompts = {
        "Calorie in an apple?": "What are the calories in an apple?",
        "Gimme some snacks.": "Suggest some healthy snack ideas", # Changed key to match button text
        "What is fiber?": "Why is fiber good for you?"
    }
    button_clicked_prompt = None
    # Using user's button text as labels
    if cols_ex[0].button("Calorie in an apple?", key="ex_btn_1"):
        button_clicked_prompt = example_prompts["Calorie in an apple?"]
        st.session_state.chat_button_clicked = True
    if cols_ex[1].button("Gimme some snacks.", key="ex_btn_2"): # Changed button text
         button_clicked_prompt = example_prompts["Gimme some snacks."]
         st.session_state.chat_button_clicked = True
    if cols_ex[2].button("What is fiber?", key="ex_btn_3"):
         button_clicked_prompt = example_prompts["What is fiber?"]
         st.session_state.chat_button_clicked = True

    # Handle button click immediately
    if button_clicked_prompt:
        handle_chat_query(button_clicked_prompt)

    # Display latest Q&A pair
    if st.session_state.chat_latest_query:
        # REMOVED key argument
        with st.chat_message("user"):
            st.markdown(st.session_state.chat_latest_query)
    if st.session_state.chat_latest_response:
         # REMOVED key argument
        with st.chat_message("assistant"):
            st.markdown(st.session_state.chat_latest_response)
    # Show initial message only if FAQ/model loaded successfully and no chat started
    elif faq_load_successful and not st.session_state.chat_latest_query:
         # REMOVED key argument
         with st.chat_message("assistant"):
              # Applying user's initial message change
              st.info("How can I help with your nutrition today?")

    # Chat input box
    user_input = st.chat_input("Ask a nutrition question...", key="chat_input_main")
    if user_input:
        handle_chat_query(user_input)

    # Add Clear Chat Button
    if st.button("Clear Chat History", key="clear_chat_btn"):
        st.session_state.chat_latest_query = None
        st.session_state.chat_latest_response = None
        st.rerun()

    # Close the chatbot container div AFTER all elements inside it - Restored
    st.markdown('</div>', unsafe_allow_html=True)


# --- Footer / Creators Section ---
st.markdown("---")

# Applying user's expander title change
with st.expander("✨ Meet My Creators ✨"):
    # Apply the container class using markdown BEFORE elements - Restored
    # st.markdown('<div class="creators-container">', unsafe_allow_html=True)
    col_creator1, col_creator2 = st.columns(2)
    with col_creator1:
        # Applying user's text/details changes
        st.markdown("**Kanishk Aman** (B.Tech, 3rd Year)")
        st.write("*Indian Institute of Science, Bangalore*")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/kanishk-aman/)")
        st.caption("Data & Backend")
    with col_creator2:
        # Applying user's text/details changes
        st.markdown("**Jenifer Maibam** (B.Tech, 2nd Year)")
        st.write("*NIAMT (Formerly NIFFT), Ranchi*")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/jenifer-maibam-a8413b294/)")
        st.caption("Frontend & UI/UX")
    # Close the creators container div - Restored
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("Nutritional data is approximate per serving. Recipe ratings are from Epicurious users.")
# User's final markdown line is kept commented out as in their version
# st.markdown("Made with ❤️ by Kanishk & Jenifer")

