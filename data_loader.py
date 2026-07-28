import pandas as pd
import json
import numpy as np

def parse_genres(genre_str):
    """
    Parse genres from string. Supports both JSON format (TMDB) and simple strings.
    """
    if not isinstance(genre_str, str):
        return ["Unknown"]
    
    genre_str = genre_str.strip()
    if not genre_str:
        return ["Unknown"]
        
    # Check if it looks like JSON array
    if genre_str.startswith("[") and genre_str.endswith("]"):
        try:
            genres_list = json.loads(genre_str)
            names = [g['name'] for g in genres_list if 'name' in g]
            return names if names else ["Unknown"]
        except Exception:
            pass
            
    # Try parsing comma-separated names
    names = [g.strip() for g in genre_str.split(",") if g.strip()]
    return names if names else ["Unknown"]

def load_and_clean_data(file_path):
    """
    Load and clean the movie dataset.
    Returns:
        df: Cleaned dataframe
        all_genres: A list of all unique genres in the dataset
    """
    df = pd.read_csv(file_path)
    
    # Handle column naming variations (e.g. vote_aver -> vote_average)
    if 'vote_aver' in df.columns and 'vote_average' not in df.columns:
        df = df.rename(columns={'vote_aver': 'vote_average'})
        
    # Handle missing vote_count (simulate if missing in uploaded file)
    if 'vote_count' not in df.columns:
        # Create a synthetic vote count based on popularity
        df['vote_count'] = (df['popularity'] * 50).fillna(10).astype(int) + 10
        
    # Required columns check
    required_cols = ['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'vote_count', 'genres']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")
            
    # Ensure numeric types
    numeric_cols = ['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'vote_count']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop rows with null values in core numeric columns
    df = df.dropna(subset=numeric_cols)
    
    # Clean budget and revenue (budget must be > 0, revenue can be >= 0)
    df = df[(df['budget'] > 0) & (df['revenue'] >= 0)]
    df = df.dropna(subset=numeric_cols) # recheck after numeric conversion
    
    # Parse genres
    df['parsed_genres'] = df['genres'].apply(parse_genres)
    
    # Define primary genre
    df['primary_genre'] = df['parsed_genres'].apply(lambda x: x[0] if x else "Unknown")
    
    # Create the binary target: success (Revenue > Budget)
    df['success'] = (df['revenue'] > df['budget']).astype(int)
    
    # Collect all unique genres
    all_genres_set = set()
    for g_list in df['parsed_genres']:
        all_genres_set.update(g_list)
        
    # Remove 'Unknown' from unique genres list if there are others
    all_genres = sorted(list(all_genres_set))
    if "Unknown" in all_genres and len(all_genres) > 1:
        all_genres.remove("Unknown")
        all_genres.append("Unknown")
        
    return df, all_genres

