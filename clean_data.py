import pandas as pd
import os

# --- Configuration ---
# Define the path to the raw data file relative to this script
RAW_DATA_FILE = 'epi_r.csv'
# Define the path where the cleaned data will be saved
CLEANED_DATA_FILE = 'recipes_cleaned.csv'
# Define essential columns needed for the app
ESSENTIAL_COLUMNS = ['title', 'rating', 'calories', 'protein', 'fat', 'sodium']
# Define potential dietary/tag columns to keep (adjust based on exact column names in your file)
# Check your epi_r.csv for the exact names (e.g., 'glutenFree' vs 'gluten free')
TAG_COLUMNS = [
    'vegetarian', 'vegan', 'glutenFree', 'dairyFree', 'peanutFree',
    'soyFree', 'treeNutFree', 'kosher', 'healthy', 'lowCal', 'lowFat',
    'lowSodium', 'dessert', 'drink' # Add other relevant tags if needed
]


# --- Main Cleaning Function ---
def clean_recipe_data(input_filepath, output_filepath):
    """
    Loads, cleans, and saves the recipe data.
    """
    print(f"Attempting to load data from: {input_filepath}")
    try:
        # Load the raw dataset
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded {len(df)} rows.")

        # --- 1. Select Relevant Columns ---
        # Check which tag columns actually exist in the loaded dataframe
        existing_tag_columns = [col for col in TAG_COLUMNS if col in df.columns]
        columns_to_keep = ESSENTIAL_COLUMNS + existing_tag_columns
        
        # Ensure 'title' exists, as it's crucial
        if 'title' not in df.columns:
            print("Error: 'title' column not found in the dataset. Cannot proceed.")
            return

        # Keep only the necessary columns + existing tags
        df = df[columns_to_keep]
        print(f"Selected relevant columns: {list(df.columns)}")

        # --- 2. Handle Missing Values ---
        # Drop rows where essential nutritional info or title is missing
        print(f"Rows before dropping NaNs in essentials: {len(df)}")
        df = df.dropna(subset=ESSENTIAL_COLUMNS)
        print(f"Rows after dropping NaNs in essentials: {len(df)}")

        # Fill NaNs in tag columns with 0 (assuming missing means 'false')
        for col in existing_tag_columns:
             df[col] = df[col].fillna(0)

        # --- 3. Ensure Correct Data Types ---
        # Convert nutritional columns and rating to numeric (float)
        for col in ['rating', 'calories', 'protein', 'fat', 'sodium']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors will turn problematic values into NaN

        # Convert tag columns to integers (0 or 1)
        for col in existing_tag_columns:
             # Check if column is already numeric-like before converting
             if pd.api.types.is_numeric_dtype(df[col]):
                 df[col] = df[col].astype(int)
             else:
                 # Handle potential non-numeric entries if necessary (e.g., 'True'/'False' strings)
                 # This basic conversion assumes they are mostly 0/1 or NaN already
                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)


        # Drop rows again if numeric conversion failed for essential columns
        print(f"Rows before dropping NaNs after type conversion: {len(df)}")
        df = df.dropna(subset=ESSENTIAL_COLUMNS)
        print(f"Rows after dropping NaNs after type conversion: {len(df)}")

        # --- 4. Remove Outliers/Invalid Data ---
        print(f"Rows before removing invalid data: {len(df)}")
        # Remove recipes with 0 or negative calories, or excessively high values (adjust thresholds as needed)
        df = df[df['calories'] > 10] # At least 10 calories
        df = df[df['calories'] < 5000] # Less than 5000 calories (adjust as needed)
        # Ensure rating is within a sensible range (e.g., 0-5)
        if 'rating' in df.columns:
            df = df[(df['rating'] >= 0) & (df['rating'] <= 5)]
        print(f"Rows after removing invalid data: {len(df)}")

        # --- 5. Save Cleaned Data ---
        if not df.empty:
            # Construct the full output path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_output_path = os.path.join(script_dir, 'data', output_filepath) # Save to data folder
            
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(full_output_path), exist_ok=True) 
            
            df.to_csv(full_output_path, index=False)
            print(f"Cleaned data saved successfully to: {full_output_path}")
        else:
            print("Warning: No data left after cleaning. Cleaned file not saved.")

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {input_filepath}")
    except Exception as e:
        print(f"An error occurred during data cleaning: {e}")
        import traceback
        traceback.print_exc()


# --- Execution ---
if __name__ == "__main__":
    # Assuming epi_r.csv is in the same directory as this script OR in a 'data' subdirectory
    # Try finding the file in common locations relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, RAW_DATA_FILE),
        os.path.join(script_dir, 'data', RAW_DATA_FILE), # If script is in 'models' or similar
        RAW_DATA_FILE # If script is run from the project root
    ]
    
    input_path_found = None
    for path in possible_paths:
        if os.path.exists(path):
            input_path_found = path
            break
            
    if input_path_found:
         clean_recipe_data(input_path_found, CLEANED_DATA_FILE)
    else:
        print(f"Error: Could not find the raw data file '{RAW_DATA_FILE}' in expected locations.")

