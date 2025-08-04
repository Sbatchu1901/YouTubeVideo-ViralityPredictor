# File: 3_analysis/exploratory_data_analysis.py
# This script performs basic exploratory data analysis on the cleaned dataset.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_explore_data(file_path):
    """
    Loads a CSV file and performs initial data exploration.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    # Check if the file exists before attempting to load
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at '{file_path}'. Please check the file path.")
        return None

    df = pd.read_csv(file_path)

    print(" Data loaded successfully.")
    print("\n--- First 5 rows of the DataFrame ---")
    print(df.head())
    
    print("\n--- DataFrame Information ---")
    df.info()
    
    print("\n--- Summary Statistics of Numerical Columns ---")
    print(df.describe())
    
    return df

def visualize_data(df):
    """
    Generates and displays key visualizations for the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to visualize.
    """
    if df is None:
        return

    # Set up the plotting style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    fig.suptitle('Expanded Exploratory Data Analysis of YouTube Video Features', fontsize=20)

    # Visualization 1: Distribution of the Viral Label
    sns.countplot(x='viral_label', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Viral Videos (1: Viral, 0: Not Viral)')
    axes[0, 0].set_xlabel('Viral Label')
    axes[0, 0].set_ylabel('Number of Videos')
    axes[0, 0].set_xticks(ticks=[0, 1], labels=['Not Viral', 'Viral'])

    # Visualization 2: Correlation between Views and Likes
    sns.scatterplot(x='views', y='likes', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Views vs. Likes')
    axes[0, 1].set_xlabel('Views')
    axes[0, 1].set_ylabel('Likes')

    # Visualization 3: Views per day for Viral vs. Non-Viral Videos
    sns.boxplot(x='viral_label', y='views_per_day', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Views Per Day by Viral Label')
    axes[1, 0].set_xlabel('Viral Label')
    axes[1, 0].set_ylabel('Views Per Day')
    axes[1, 0].set_xticks(ticks=[0, 1], labels=['Not Viral', 'Viral'])

    # Visualization 4: Mean Toxicity for Viral vs. Non-Viral Videos
    sns.boxplot(x='viral_label', y='mean_toxicity', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Mean Comment Toxicity by Viral Label')
    axes[1, 1].set_xlabel('Viral Label')
    axes[1, 1].set_ylabel('Mean Toxicity Score')
    axes[1, 1].set_xticks(ticks=[0, 1], labels=['Not Viral', 'Viral'])

    # Visualization 5: Distribution of Title Length
    sns.histplot(df['title_len'], kde=True, ax=axes[2, 0])
    axes[2, 0].set_title('Distribution of Title Lengths')
    axes[2, 0].set_xlabel('Title Length')
    axes[2, 0].set_ylabel('Frequency')

    # Visualization 6: Correlation Heatmap of all Numerical Features
    corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2, 1])
    axes[2, 1].set_title('Correlation Matrix of Numerical Features')

    # Adjust layout to prevent titles from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Display the plots
    plt.show()

if __name__ == '__main__':
    # Define the path to your cleaned CSV file using forward slashes
    file_path = "C:/Users/sruja/OneDrive/Desktop/YOUTUBE/2_preprocessing/clean_data.csv"
    # Load the data
    youtube_df = load_and_explore_data(file_path)
    
    # If the data was loaded successfully, proceed with visualization
    if youtube_df is not None:
        visualize_data(youtube_df)
