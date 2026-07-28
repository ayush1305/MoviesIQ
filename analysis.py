import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

# Strict Color Palette (1-2 colors)
PRIMARY_COLOR = '#1f4068'  # Deep Slate Blue
ACCENT_COLOR = '#16a085'   # Cool Teal
BG_COLOR = '#f8f9fa'       # Very Light Gray

def set_style():
    """
    Set custom style for matplotlib to look clean and professional.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def plot_budget_vs_revenue(df):
    """
    1. Plot Budget vs. Revenue using a scatter plot.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Use different markers or shades for success/failure with the 2-color palette
    sns.scatterplot(
        data=df, 
        x='budget', 
        y='revenue', 
        hue='success',
        palette={0: '#95a5a6', 1: PRIMARY_COLOR, '0': '#95a5a6', '1': PRIMARY_COLOR},
        alpha=0.6,
        ax=ax,
        edgecolor=None
    )
    
    # Scale axes to Millions for readability
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}M".format(int(x/1e6))))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}M".format(int(x/1e6))))
    
    ax.set_title('Movie Budget vs. Revenue', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Budget (USD, Millions)', fontsize=10)
    ax.set_ylabel('Revenue (USD, Millions)', fontsize=10)
    
    # Add diagonal break-even line (Revenue = Budget)
    max_val = max(df['budget'].max(), df['revenue'].max())
    ax.plot([0, max_val], [0, max_val], color='#e74c3c', linestyle='--', alpha=0.5, label='Break-even Line')
    
    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ['Unsuccessful', 'Successful', 'Break-even']
    ax.legend(handles, new_labels, title="Outcome", loc="upper left", frameon=True)
    
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

def plot_genre_trends(df):
    """
    2. Explore genre trends: which genres are most common, and which tend to be most successful?
    """
    set_style()
    
    # Explode the parsed_genres column to get one row per genre per movie
    exploded_df = df.explode('parsed_genres')
    
    # Calculate counts
    genre_counts = exploded_df['parsed_genres'].value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    
    # Calculate success rate per genre
    genre_success = exploded_df.groupby('parsed_genres')['success'].mean().reset_index()
    genre_success.columns = ['Genre', 'Success Rate']
    
    # Merge
    genre_data = pd.merge(genre_counts, genre_success, on='Genre').sort_values(by='Count', ascending=False).head(10)
    
    # Visualise both side-by-side using matplotlib/seaborn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Most Common Genres (using PRIMARY_COLOR)
    sns.barplot(
        data=genre_data,
        y='Genre',
        x='Count',
        color=PRIMARY_COLOR,
        ax=ax1,
        edgecolor=None
    )
    ax1.set_title('Top 10 Most Common Genres', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Number of Movies', fontsize=10)
    ax1.set_ylabel('', fontsize=10)
    
    # Plot 2: Success Rate of those top genres (using ACCENT_COLOR)
    genre_data_sorted_success = genre_data.sort_values(by='Success Rate', ascending=False)
    sns.barplot(
        data=genre_data_sorted_success,
        y='Genre',
        x='Success Rate',
        color=ACCENT_COLOR,
        ax=ax2,
        edgecolor=None
    )
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x*100)}%"))
    ax2.set_title('Success Rate of Common Genres', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Success Rate (%)', fontsize=10)
    ax2.set_ylabel('', fontsize=10)
    
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

def plot_success_associations(df):
    """
    3. Examine how popularity, runtime, and vote_average relate to success.
    """
    set_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    features = ['popularity', 'runtime', 'vote_average']
    titles = ['Popularity vs Success', 'Runtime vs Success', 'Vote Average vs Success']
    x_labels = ['Popularity Score', 'Runtime (Minutes)', 'Vote Average']
    
    for i, feature in enumerate(features):
        ax = axes[i]
        sns.boxplot(
            data=df,
            x='success',
            y=feature,
            palette={0: '#95a5a6', 1: PRIMARY_COLOR, '0': '#95a5a6', '1': PRIMARY_COLOR},
            ax=ax,
            width=0.5,
            showfliers=False # Hide extreme outliers to see the quartiles clearly
        )
        ax.set_xticklabels(['Unsuccessful', 'Successful'])
        ax.set_title(titles[i], fontsize=11, fontweight='bold', pad=12)
        ax.set_xlabel('')
        ax.set_ylabel(x_labels[i], fontsize=9)
        
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    """
    4. Produce a correlation heatmap of the numeric features.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    numeric_cols = ['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'vote_count']
    corr_matrix = df[numeric_cols].corr()
    
    # Custom 2-color diverging palette (Light gray -> Primary Color)
    cmap = sns.diverging_palette(220, 240, as_cmap=True)
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": .8},
        ax=ax
    )
    
    ax.set_title('Correlation Heatmap of Numeric Features', fontsize=12, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

def run_t_test(df, feature='popularity'):
    """
    Stage 3: T-Test
    Check whether a numeric feature differs significantly between successful and unsuccessful movies.
    """
    group_success = df[df['success'] == 1][feature]
    group_fail = df[df['success'] == 0][feature]
    
    # Run independent t-test (Welch's t-test as variances are likely unequal)
    t_stat, p_val = stats.ttest_ind(group_success, group_fail, equal_var=False)
    
    null_hypothesis = f"The mean {feature} of successful movies is equal to that of unsuccessful movies."
    alternative_hypothesis = f"The mean {feature} of successful movies differs from that of unsuccessful movies."
    
    conclusion = ""
    if p_val < 0.05:
        conclusion = f"Reject the null hypothesis. There is a statistically significant difference in mean {feature} between successful and unsuccessful movies (p < 0.05)."
    else:
        conclusion = f"Fail to reject the null hypothesis. There is no statistically significant difference in mean {feature} between successful and unsuccessful movies (p >= 0.05)."
        
    return {
        'feature': feature,
        'null_hypothesis': null_hypothesis,
        'alternative_hypothesis': alternative_hypothesis,
        't_statistic': t_stat,
        'p_value': p_val,
        'conclusion': conclusion
    }

def run_chi_square_test(df):
    """
    Stage 3: Chi-Square Test
    Check whether a categorical feature (genre) is associated with success.
    """
    # Explode the parsed_genres to assign genre labels per movie row
    exploded_df = df.explode('parsed_genres')
    
    # Create contingency table
    contingency_table = pd.crosstab(exploded_df['parsed_genres'], exploded_df['success'])
    
    # Run test
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    
    null_hypothesis = "Movie genre is independent of movie success (no association between genre and success)."
    alternative_hypothesis = "Movie genre is associated with movie success (genre affects the likelihood of success)."
    
    conclusion = ""
    if p_val < 0.05:
        conclusion = "Reject the null hypothesis. Movie genre is significantly associated with movie success (p < 0.05)."
    else:
        conclusion = "Fail to reject the null hypothesis. Movie genre is independent of movie success (p >= 0.05)."
        
    return {
        'null_hypothesis': null_hypothesis,
        'alternative_hypothesis': alternative_hypothesis,
        'chi2_statistic': chi2,
        'p_value': p_val,
        'degrees_of_freedom': dof,
        'conclusion': conclusion
    }
