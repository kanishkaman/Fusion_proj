# Meal Planner & Nutrition Assistant

A Streamlit web application designed to help users, especially students, plan healthy meals based on preferences, get nutritional information, and reduce food stress.

## ✨ Features

* **Personalized Recipe Recommendations:** Suggests recipes based on user-defined preferences:
    * Calorie range (Low, Medium, High)
    * Protein range (Low, Medium, High)
    * Dietary tags (Vegetarian, Vegan, Gluten-Free, etc. - based on available data)
    * Keyword search (e.g., "chicken", "quick pasta")
* **NutriBot Assistant:** An interactive chatbot (powered by Sentence Transformers and a curated FAQ dataset) to answer questions about:
    * Nutrition facts (calories, macros in common foods)
    * Ingredient substitutions
    * Basic cooking tips
    * Definitions of dietary terms
* **Dynamic Filtering:** Recommendations update based on sidebar selections.
* **Responsive UI:** Built with Streamlit for web access.

## 🛠️ Tech Stack

* **Language:** Python 3
* **Web Framework:** Streamlit
* **Data Handling:** Pandas
* **Machine Learning:**
    * Scikit-learn (TF-IDF for keyword matching)
    * Sentence Transformers (for semantic search in chatbot)
    * Torch (dependency for Sentence Transformers)
             
            
## 🚀 Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/Fusion_proj.git
    cd Fusion_proj
    ```
2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    # On Windows:
    # .venv\Scripts\activate
    # On macOS/Linux:
    # source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The first time you run the app, the Sentence Transformer model will be downloaded automatically, which might take a few minutes).*
4.  **(Optional) Generate Cleaned Data:** If `recipes_cleaned.csv` is not included or you want to regenerate it, ensure you have the raw data (e.g., `epi_r.csv` from Kaggle) and the `clean_data.py` script. Run the cleaning script:
    ```bash
    python path/to/clean_data.py
    ```
    *(Ensure the script correctly finds the raw data and saves the cleaned file to the `data/` directory).*
5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
6.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

