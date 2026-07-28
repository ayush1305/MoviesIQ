import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Strict Color Palette (1-2 colors)
PRIMARY_COLOR = '#1f4068'  # Deep Slate Blue
ACCENT_COLOR = '#16a085'   # Cool Teal

def prepare_data(df):
    """
    Prepare features and target for training.
    """
    # Select feature columns
    feature_cols = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'primary_genre']
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df['success'].copy()
    
    # Handle categorical variables (primary_genre)
    # Get dummies for genres
    X = pd.get_dummies(X, columns=['primary_genre'], prefix='genre', drop_first=True)
    
    return X, y

def train_and_evaluate_model(df, test_size=0.2, random_state=42):
    """
    Split data, train Random Forest, evaluate metrics and feature importance.
    """
    # Prepare features and target
    X, y = prepare_data(df)
    
    # Split into train and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = rf_model.predict(X_test)
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature Importances
    importances = rf_model.feature_importances_
    features_list = X.columns
    feature_imp_df = pd.DataFrame({
        'Feature': features_list,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Train on full data for deploying in inference
    final_model = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)
    final_model.fit(X, y)
    
    # Save the list of dummy feature columns for inference alignment
    feature_columns_list = list(X.columns)
    
    return {
        'model': final_model,
        'feature_columns': feature_columns_list,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'feature_importance': feature_imp_df,
        'test_set_size': len(y_test),
        'train_set_size': len(y_train)
    }

def plot_confusion_matrix(cm):
    """
    Generate a clean Matplotlib plot for the Confusion Matrix.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Use 2-color palette for heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap=sns.light_palette(PRIMARY_COLOR, as_cmap=True),
        cbar=False,
        square=True,
        ax=ax,
        xticklabels=['Unsuccessful', 'Successful'],
        yticklabels=['Unsuccessful', 'Successful']
    )
    
    ax.set_title('Confusion Matrix', fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=9)
    ax.set_ylabel('True Label', fontsize=9)
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_imp_df):
    """
    Generate a clean horizontal bar plot for feature importances.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Take the top 10 features for display clarity
    top_10 = feature_imp_df.head(10)
    
    # Clean feature names (remove "genre_")
    top_10 = top_10.copy()
    top_10['CleanFeature'] = top_10['Feature'].apply(lambda x: x.replace('genre_', 'Genre: ') if x.startswith('genre_') else x.capitalize())
    
    sns.barplot(
        data=top_10,
        y='CleanFeature',
        x='Importance',
        color=PRIMARY_COLOR,
        ax=ax,
        edgecolor=None
    )
    
    ax.set_title('Top 10 Feature Importances', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Relative Importance', fontsize=10)
    ax.set_ylabel('', fontsize=10)
    
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

def predict_single(model, feature_columns, user_inputs):
    """
    Predict success for a single movie based on user inputs.
    user_inputs is a dict: {
        'budget': val,
        'popularity': val,
        'runtime': val,
        'vote_average': val,
        'vote_count': val,
        'primary_genre': val
    }
    """
    # Create empty row representing features
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Set numeric values
    input_df['budget'] = user_inputs['budget']
    input_df['popularity'] = user_inputs['popularity']
    input_df['runtime'] = user_inputs['runtime']
    input_df['vote_average'] = user_inputs['vote_average']
    input_df['vote_count'] = user_inputs['vote_count']
    
    # Set one-hot genre if it exists in columns
    genre_col = f"genre_{user_inputs['primary_genre']}"
    if genre_col in input_df.columns:
        input_df[genre_col] = 1
        
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    return prediction, probabilities[1]
