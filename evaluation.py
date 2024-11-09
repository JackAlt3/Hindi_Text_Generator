import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# Load the dataset from a CSV file
def load_dataset(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    return df

# Function to create n-gram model from the Hindi sentences
def create_ngram_model(df, n=3):
    ngrams = defaultdict(Counter)
    for index, row in df.iterrows():
        words = row['hindi_sentence'].split()
        for i in range(len(words) - n + 1):
            prefix = tuple(words[i:i+n-1])
            next_word = words[i+n-1]
            ngrams[prefix][next_word] += 1
    return ngrams

# Function to calculate perplexity of the model
def calculate_perplexity(ngrams, test_sentences, n=3):
    total_log_prob = 0
    N = 0
    vocab_size = len(set(word for prefix in ngrams for word in ngrams[prefix]))

    for sentence in test_sentences:
        words = sentence.split()
        for i in range(len(words) - n + 1):
            prefix = tuple(words[i:i+n-1])
            next_word = words[i+n-1]
            if prefix in ngrams:
                total_count = sum(ngrams[prefix].values())
                word_count = ngrams[prefix][next_word]
                # Laplace smoothing
                probability = (word_count + 1) / (total_count + vocab_size)
                log_prob = np.log2(probability) if probability > 0 else float('-inf')
                total_log_prob += log_prob
                N += 1
            else:
                # If the prefix is not found, consider the probability as very small
                total_log_prob += np.log2(1 / (vocab_size + 1))  # Smoothing for unseen prefix
                N += 1

    perplexity = 2 ** (-total_log_prob / N) if N > 0 else float('inf')
    return perplexity

# Main function to execute the evaluation
def evaluate_model():
    # Load the dataset
    file_path = 'sentences.csv'  # Specify the dataset file path here
    df = load_dataset(file_path)
    
    # Split data into training and testing sets (80-20 split)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    # Create n-gram models and calculate perplexity for different n
    n_values = [2, 3, 4]
    perplexities = []

    for n in n_values:
        ngrams = create_ngram_model(train_df, n)
        perplexity = calculate_perplexity(ngrams, test_df['hindi_sentence'], n)
        perplexities.append(perplexity)
        print(f'Perplexity for n={n}: {perplexity}')

    # Plotting the perplexity values
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, perplexities, marker='o')
    plt.title('Perplexity of N-gram Models')
    plt.xlabel('N (N-gram size)')
    plt.ylabel('Perplexity')
    plt.xticks(n_values)
    plt.grid()
    plt.show()

# Execute the evaluation
if __name__ == "__main__":
    evaluate_model()
