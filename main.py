import pandas as pd
from collections import defaultdict, Counter
import tkinter as tk
from tkinter import scrolledtext, messagebox

# Load the dataset from a CSV file with error handling
def load_dataset(file_path):
    try:
        # Specify delimiter if needed (comma, semicolon, etc.)
        df = pd.read_csv(file_path, encoding='utf-8', quotechar='"', delimiter=',')
        return df
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        messagebox.showerror("CSV Error", f"Error reading the CSV file: {e}")
        return None  # Return None if there's an error

# Function to create n-gram model from the Hindi sentences (n can be 3 for trigram or 2 for bigram)
def create_ngram_model(df, n=3):
    ngrams = defaultdict(Counter)  # Use Counter for frequency counting
    for index, row in df.iterrows():
        words = row['hindi_sentence'].split()  # Split Hindi sentence into words
        for i in range(len(words) - n + 1):
            prefix = tuple(words[i:i+n-1])  # Get the prefix of size n-1
            next_word = words[i+n-1]  # The word that follows the prefix
            ngrams[prefix][next_word] += 1  # Add to frequency count
    return ngrams

# Function to pick the most frequent next word based on frequency
def most_frequent_choice(counter):
    return counter.most_common(1)[0][0]  # Get the word with the highest frequency

# Function to detect if a sentence is complete (using "।" as a full stop)
def is_sentence_complete(word):
    return word == '।'

# Function to generate new Hindi text based on the previous words
def generate_hindi_text(ngrams, input_text, complete_sentence=False, max_length=7, n=3):
    words = input_text.split()
    generated_sentence = words[:]  # Start with the input words

    # Continue generating words until the sentence is complete or max length is reached
    while True:
        if len(generated_sentence) < n-1:
            break  # Stop if the input is shorter than needed for the n-gram model

        prefix = tuple(generated_sentence[-(n-1):])  # Use the last (n-1) words as context
        if prefix in ngrams:
            next_word = most_frequent_choice(ngrams[prefix])
            generated_sentence.append(next_word)

            if is_sentence_complete(next_word):
                break
        else:
            if len(generated_sentence) > 1:
                bigram_prefix = tuple(generated_sentence[-2:])
                if bigram_prefix in ngrams:
                    next_word = most_frequent_choice(ngrams[bigram_prefix])
                    generated_sentence.append(next_word)

                    if is_sentence_complete(next_word):
                        break
                else:
                    unigram_prefix = (generated_sentence[-1],)
                    if unigram_prefix in ngrams:
                        next_word = most_frequent_choice(ngrams[unigram_prefix])
                        generated_sentence.append(next_word)

                        if is_sentence_complete(next_word):
                            break
                    else:
                        break  # No valid prefix found, stop generation
            else:
                break  # Stop if no valid next word is found

        # If not generating the full sentence and max length is reached, break the loop
        if not complete_sentence and len(generated_sentence) >= max_length:
            break

    return ' '.join(generated_sentence)

# Main class for the Hindi Text Generator application with GUI
class HindiTextGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hindi Text Generator")
        self.root.geometry("450x400")

        # Load the dataset
        self.file_path = 'sentences.csv'  # Specify the dataset file path here
        self.df = load_dataset(self.file_path)

        if self.df is None:
            return  # If loading the dataset failed, exit the app

        # Create n-gram model
        self.ngrams = create_ngram_model(self.df, n=3)  # Using trigram for better prediction

        # Input label
        self.label = tk.Label(root, text="Enter a Hindi sentence:")
        self.label.pack(pady=10)

        # Text input field
        self.input_text = tk.Entry(root, width=50)
        self.input_text.pack(pady=10)

        # Toggle for generating complete sentence or a few words
        self.complete_sentence_var = tk.BooleanVar()
        self.complete_sentence_toggle = tk.Checkbutton(root, text="Generate complete sentence", variable=self.complete_sentence_var)
        self.complete_sentence_toggle.pack(pady=5)

        # Generate button
        self.generate_button = tk.Button(root, text="Generate Text", command=self.generate_text)
        self.generate_button.pack(pady=5)

        # Output text area
        self.output_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10)
        self.output_area.pack(pady=10)

    def generate_text(self):
        input_sentence = self.input_text.get().strip()
        if not input_sentence:
            messagebox.showwarning("Input Error", "Please enter a Hindi sentence.")
            return

        # Determine if the user wants to generate a complete sentence or a limited length
        complete_sentence = self.complete_sentence_var.get()

        # Generate Hindi text
        generated_text = generate_hindi_text(self.ngrams, input_sentence, complete_sentence=complete_sentence, max_length=7)
        self.output_area.delete(1.0, tk.END)  # Clear previous output
        self.output_area.insert(tk.END, generated_text)  # Insert the generated text

# Execute the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = HindiTextGeneratorApp(root)
    root.mainloop()
