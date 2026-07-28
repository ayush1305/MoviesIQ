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

def generate_mock_data():
    """
    Generate mock TMDB-like dataset if the user hasn't uploaded their own.
    This lets the app run out-of-the-box.
    """
    np.random.seed(42)
    n_samples = 500
    
    genres_pool = ["Action", "Adventure", "Fantasy", "Science Fiction", "Drama", "Comedy", "Thriller", "Romance", "Horror"]
    
    budgets = np.random.exponential(scale=50000000, size=n_samples) + 1000000
    # Higher budget generally correlates with higher revenue, but with high variance
    revenues = budgets * np.random.lognormal(mean=0.3, sigma=0.8, size=n_samples)
    
    popularity = np.random.exponential(scale=20, size=n_samples) + 1.0
    runtimes = np.random.normal(loc=110, scale=20, size=n_samples)
    runtimes = np.clip(runtimes, 60, 200)
    
    vote_averages = np.random.normal(loc=6.2, scale=1.0, size=n_samples)
    vote_averages = np.clip(vote_averages, 1.0, 10.0)
    
    vote_counts = (popularity * np.random.poisson(lam=50, size=n_samples)).astype(int) + 10
    
    titles = [f"Mock Movie {i}" for i in range(n_samples)]
    
    # Generate mock genres JSON strings
    mock_genres_col = []
    for _ in range(n_samples):
        n_genres = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
        selected = np.random.choice(genres_pool, size=n_genres, replace=False)
        genres_list = [{"id": i, "name": g} for i, g in enumerate(selected)]
        mock_genres_col.append(json.dumps(genres_list))
        
    mock_df = pd.DataFrame({
        'title': titles,
        'budget': budgets.astype(int),
        'revenue': revenues.astype(int),
        'popularity': popularity,
        'runtime': runtimes.astype(int),
        'vote_average': vote_averages,
        'vote_count': vote_counts,
        'genres': mock_genres_col
    })
    
    return mock_df
