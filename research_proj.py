import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from nltk import ngrams

# first attempt without parsing

# Read the DataFrame from the pickle file
df = pd.read_pickle("reviews_segment.pkl")

# Preprocess review function
def preprocess_review(review):
    # Convert to lowercase
    review = review.lower()
    review = re.sub(r'[^a-z\s]', '', review)
    stop_words = set(stopwords.words('english'))
    review = ' '.join(word for word in review.split() if word not in stop_words)
    return review

# Function to generate n-grams
def generate_ngrams(text, n):
    tokens = text.split()
    ngrams_list = list(ngrams(tokens, n))
    return [' '.join(gram) for gram in ngrams_list]

# Example list of predefined aspects and opinions
aspects = ['battery life', 'sound quality', 'screen', 'camera', 'price', 'durability']
positive_opinions = ['good', 'great', 'excellent', 'amazing', 'fantastic']
negative_opinions = ['bad', 'poor', 'terrible', 'horrible', 'disappointing']

# New function to find aspect-opinion pairs in the review using multiple n-grams
def extract_aspect_opinion_with_multiple_ngrams(cleaned_review):
    found_aspects = set()  # Use a set to store unique aspects
    found_opinions = set()  # Use a set to store unique opinions
    
    # Check for unigrams, bigrams, and trigrams in the review
    for n in range(1, 4):  # Check for unigrams (1), bigrams (2), trigrams (3)
        ngrams_list = generate_ngrams(cleaned_review, n)
        
        # Look for aspects and opinions
        for gram in ngrams_list:
            for aspect in aspects:
                if aspect in gram:
                    found_aspects.add(aspect)  # Add to set (no duplicates)
            for opinion in positive_opinions + negative_opinions:
                if opinion in gram:
                    found_opinions.add(opinion)  # Add to set (no duplicates)
    
    # Convert sets back to lists to keep original structure
    return list(found_aspects), list(found_opinions)

# Apply preprocessing to the 'review_text' column
df['cleaned_review_text'] = df['review_text'].apply(preprocess_review)

# Apply the new extraction function to the DataFrame
df['aspects_opinions'] = df['cleaned_review_text'].apply(extract_aspect_opinion_with_multiple_ngrams)

# Display the results
print(df[['cleaned_review_text', 'aspects_opinions']].head())
