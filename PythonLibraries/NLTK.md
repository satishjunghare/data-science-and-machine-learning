# NLTK Library

## Overview
NLTK (Natural Language Toolkit) is a comprehensive Python library for natural language processing (NLP). It provides easy-to-use interfaces to over 50 corpora and lexical resources, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning. NLTK is widely used in academia and industry for research and development in computational linguistics and NLP.

## Installation
```bash
# Basic installation
pip install nltk

# Latest version
pip install nltk==3.8.1

# Download NLTK data
python -c "import nltk; nltk.download('all')"

# Download specific packages
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Key Features
- **Text Processing**: Tokenization, stemming, lemmatization
- **Part-of-Speech Tagging**: POS tagging and chunking
- **Named Entity Recognition**: Entity detection and classification
- **Parsing**: Syntactic and semantic parsing
- **Corpora**: Access to large text collections
- **Lexical Resources**: Dictionaries, thesauri, and word lists
- **Machine Learning**: Classification and clustering tools
- **Language Models**: N-gram models and language modeling

## Core Concepts

### Text Processing
```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download required data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text
text = "Natural Language Processing (NLP) is a subfield of artificial intelligence. It helps computers understand human language. NLTK is a popular library for NLP tasks."

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:")
for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")

# Word tokenization
words = word_tokenize(text)
print(f"\nWords: {words}")

# Remove punctuation and convert to lowercase
words_clean = [word.lower() for word in words if word.isalnum()]
print(f"Clean words: {words_clean}")

# Remove stop words
stop_words = set(stopwords.words('english'))
words_no_stop = [word for word in words_clean if word not in stop_words]
print(f"Words without stop words: {words_no_stop}")

# Stemming
stemmer = PorterStemmer()
words_stemmed = [stemmer.stem(word) for word in words_no_stop]
print(f"Stemmed words: {words_stemmed}")

# Lemmatization
lemmatizer = WordNetLemmatizer()
words_lemmatized = [lemmatizer.lemmatize(word) for word in words_no_stop]
print(f"Lemmatized words: {words_lemmatized}")
```

### Part-of-Speech Tagging
```python
import nltk
from nltk import pos_tag
from nltk.chunk import RegexpParser

# Download required data
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Sample text
text = "John Smith works at Google in California. He loves programming and machine learning."

# Tokenize
tokens = nltk.word_tokenize(text)

# POS tagging
pos_tags = pos_tag(tokens)
print("POS Tags:")
for word, tag in pos_tags:
    print(f"{word}: {tag}")

# Define chunk patterns
chunk_pattern = r"""
    NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}   # Noun phrases
    VP: {<VB.*><NP|PP|CLAUSE>*}      # Verb phrases
    PP: {<IN><NP>}                   # Prepositional phrases
"""

# Create chunk parser
chunk_parser = RegexpParser(chunk_pattern)

# Parse chunks
chunks = chunk_parser.parse(pos_tags)
print("\nChunks:")
print(chunks)

# Named Entity Recognition
entities = nltk.chunk.ne_chunk(pos_tags)
print("\nNamed Entities:")
for subtree in entities:
    if hasattr(subtree, 'label'):
        print(f"{subtree.label()}: {' '.join([leaf[0] for leaf in subtree.leaves()])}")
```

### Corpus Analysis
```python
import nltk
from nltk.corpus import gutenberg, brown, reuters
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

# Download corpora
nltk.download('gutenberg')
nltk.download('brown')
nltk.download('reuters')

# Access Gutenberg corpus
emma = gutenberg.words('austen-emma.txt')
print(f"Emma text length: {len(emma)} words")

# Frequency distribution
fdist = FreqDist(emma)
print(f"Most common words: {fdist.most_common(10)}")

# Plot frequency distribution
plt.figure(figsize=(12, 6))
fdist.plot(50, cumulative=True)
plt.title('Cumulative Frequency Distribution (Emma)')
plt.show()

# Access Brown corpus
brown_words = brown.words(categories='news')
print(f"Brown news words: {len(brown_words)}")

# Genre analysis
genres = ['news', 'romance', 'humor']
for genre in genres:
    words = brown.words(categories=genre)
    fdist = FreqDist(words)
    print(f"\nMost common words in {genre}: {fdist.most_common(5)}")

# Conditional frequency distribution
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

# Plot conditional frequency
plt.figure(figsize=(12, 8))
cfd.plot(conditions=['news', 'romance', 'humor'], samples=10)
plt.title('Word Frequency by Genre')
plt.show()
```

## Text Classification

### Naive Bayes Classification
```python
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import random

# Download movie reviews corpus
nltk.download('movie_reviews')

# Prepare data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle documents
random.shuffle(documents)

# Feature extraction function
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

# Get most frequent words
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words.keys())[:2000]

# Create feature sets
featuresets = [(document_features(d), c) for (d, c) in documents]

# Split into training and testing sets
train_set, test_set = featuresets[100:], featuresets[:100]

# Train classifier
classifier = NaiveBayesClassifier.train(train_set)

# Test accuracy
print(f"Accuracy: {accuracy(classifier, test_set):.4f}")

# Show most informative features
classifier.show_most_informative_features(10)

# Classify new text
def classify_text(text):
    tokens = nltk.word_tokenize(text.lower())
    features = document_features(tokens)
    return classifier.classify(features)

# Test classification
sample_texts = [
    "This movie was absolutely fantastic!",
    "Terrible acting and boring plot.",
    "Great cinematography and excellent performances."
]

for text in sample_texts:
    sentiment = classify_text(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}\n")
```

### Sentiment Analysis
```python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import twitter_samples

# Download required data
nltk.download('vader_lexicon')
nltk.download('twitter_samples')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Sample texts
texts = [
    "I love this product! It's amazing!",
    "This is the worst thing I've ever bought.",
    "The product is okay, nothing special.",
    "Absolutely fantastic experience with great customer service!"
]

# Analyze sentiment
for text in texts:
    scores = sia.polarity_scores(text)
    print(f"Text: {text}")
    print(f"Sentiment scores: {scores}")
    
    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    print(f"Overall sentiment: {sentiment}\n")

# Analyze Twitter data
tweets = twitter_samples.strings('positive_tweets.json')[:10]
for tweet in tweets:
    scores = sia.polarity_scores(tweet)
    print(f"Tweet: {tweet[:50]}...")
    print(f"Sentiment: {scores['compound']:.3f}\n")
```

## Language Models

### N-gram Models
```python
import nltk
from nltk.util import ngrams
from nltk.probability import ConditionalFreqDist
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

# Sample text
text = "Natural language processing is a subfield of artificial intelligence. " \
       "It helps computers understand human language. " \
       "NLTK is a popular library for natural language processing tasks."

# Tokenize
tokens = nltk.word_tokenize(text.lower())

# Generate n-grams
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print("Bigrams:")
for bigram in bigrams[:10]:
    print(bigram)

print("\nTrigrams:")
for trigram in trigrams[:10]:
    print(trigram)

# Build language model
train_data, vocab = padded_everygram_pipeline(2, [tokens])
lm = MLE(2)
lm.fit(train_data, vocab)

# Generate text
def generate_text(lm, start_words, num_words=10):
    text = list(start_words)
    for _ in range(num_words):
        next_word = lm.generate(1, text_seed=text[-1:])
        text.append(next_word)
    return ' '.join(text)

# Generate text
generated = generate_text(lm, ['natural'], 15)
print(f"\nGenerated text: {generated}")

# Calculate perplexity
test_sentence = "natural language processing is useful"
test_tokens = nltk.word_tokenize(test_sentence.lower())
perplexity = lm.perplexity(test_tokens)
print(f"Perplexity: {perplexity:.2f}")
```

### Collocations
```python
import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import brown

# Get text from Brown corpus
text = brown.words(categories='news')[:10000]

# Find bigram collocations
bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(text)

# Apply frequency filter
finder.apply_freq_filter(3)

# Get collocations by different measures
print("PMI (Pointwise Mutual Information):")
pmi_collocations = finder.nbest(bigram_measures.pmi, 10)
for collocation in pmi_collocations:
    print(f"{collocation[0]} {collocation[1]}")

print("\nChi-square:")
chi_collocations = finder.nbest(bigram_measures.chi_sq, 10)
for collocation in chi_collocations:
    print(f"{collocation[0]} {collocation[1]}")

print("\nLikelihood ratio:")
lr_collocations = finder.nbest(bigram_measures.likelihood_ratio, 10)
for collocation in lr_collocations:
    print(f"{collocation[0]} {collocation[1]}")
```

## Semantic Analysis

### WordNet Integration
```python
import nltk
from nltk.corpus import wordnet as wn

# Download WordNet
nltk.download('wordnet')

# Get synsets for a word
word = "good"
synsets = wn.synsets(word)
print(f"Synsets for '{word}':")
for synset in synsets:
    print(f"  {synset.name()}: {synset.definition()}")

# Get synonyms
synonyms = []
for synset in synsets:
    for lemma in synset.lemmas():
        synonyms.append(lemma.name())
print(f"\nSynonyms for '{word}': {list(set(synonyms))}")

# Get antonyms
antonyms = []
for synset in synsets:
    for lemma in synset.lemmas():
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())
print(f"Antonyms for '{word}': {list(set(antonyms))}")

# Word similarity
word1 = "car"
word2 = "automobile"
word3 = "banana"

synset1 = wn.synsets(word1)[0]
synset2 = wn.synsets(word2)[0]
synset3 = wn.synsets(word3)[0]

similarity_12 = synset1.path_similarity(synset2)
similarity_13 = synset1.path_similarity(synset3)

print(f"\nSimilarity between '{word1}' and '{word2}': {similarity_12:.3f}")
print(f"Similarity between '{word1}' and '{word3}': {similarity_13:.3f}")

# Hypernyms and hyponyms
car_synset = wn.synsets("car")[0]
print(f"\nHypernyms of 'car':")
for hypernym in car_synset.hypernyms():
    print(f"  {hypernym.name()}: {hypernym.definition()}")

print(f"\nHyponyms of 'car':")
for hyponym in car_synset.hyponyms()[:5]:
    print(f"  {hyponym.name()}: {hyponym.definition()}")
```

### Semantic Similarity
```python
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Function to calculate semantic similarity
def semantic_similarity(sentence1, sentence2):
    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    
    tokens1 = [word.lower() for word in word_tokenize(sentence1) if word.isalnum()]
    tokens2 = [word.lower() for word in word_tokenize(sentence2) if word.isalnum()]
    
    tokens1 = [word for word in tokens1 if word not in stop_words]
    tokens2 = [word for word in tokens2 if word not in stop_words]
    
    # Get synsets for each word
    synsets1 = []
    synsets2 = []
    
    for word in tokens1:
        synsets = wn.synsets(word)
        if synsets:
            synsets1.append(synsets[0])
    
    for word in tokens2:
        synsets = wn.synsets(word)
        if synsets:
            synsets2.append(synsets[0])
    
    # Calculate similarity
    if not synsets1 or not synsets2:
        return 0.0
    
    max_similarities = []
    for synset1 in synsets1:
        similarities = []
        for synset2 in synsets2:
            similarity = synset1.path_similarity(synset2)
            if similarity:
                similarities.append(similarity)
        if similarities:
            max_similarities.append(max(similarities))
    
    if max_similarities:
        return sum(max_similarities) / len(max_similarities)
    else:
        return 0.0

# Test semantic similarity
sentences = [
    "The cat is on the mat.",
    "A feline is sitting on the carpet.",
    "The weather is nice today.",
    "It's raining outside."
]

print("Semantic Similarity Matrix:")
for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        similarity = semantic_similarity(sent1, sent2)
        print(f"{similarity:.3f}", end="\t")
    print()
```

## Use Cases
- **Text Analysis**: Document classification and sentiment analysis
- **Information Extraction**: Named entity recognition and relation extraction
- **Language Modeling**: N-gram models and text generation
- **Machine Translation**: Preprocessing and post-processing
- **Question Answering**: Text understanding and reasoning
- **Chatbots**: Natural language understanding
- **Academic Research**: Computational linguistics research
- **Content Analysis**: Social media and web content analysis

## Best Practices
1. **Data Preprocessing**: Clean and normalize text data
2. **Feature Engineering**: Extract meaningful features
3. **Model Selection**: Choose appropriate models for tasks
4. **Evaluation**: Use proper evaluation metrics
5. **Regularization**: Prevent overfitting
6. **Interpretability**: Understand model decisions
7. **Scalability**: Consider performance for large datasets
8. **Domain Adaptation**: Adapt to specific domains

## Advantages
- **Comprehensive**: Wide range of NLP tools and resources
- **Academic Quality**: Rigorous linguistic methods
- **Extensive Corpora**: Access to large text collections
- **Educational**: Excellent for learning NLP
- **Well Documented**: Comprehensive documentation
- **Active Community**: Large and active community
- **Research Oriented**: Designed for research and development
- **Free and Open**: Open source and free to use

## Limitations
- **Performance**: May be slower than modern alternatives
- **Learning Curve**: Steep learning curve for complex tasks
- **Limited Deep Learning**: Limited deep learning capabilities
- **Scalability**: May not scale well for very large datasets
- **Modern Features**: Lacks some modern NLP features
- **Production Ready**: May require additional work for production

## Related Libraries
- **spaCy**: Industrial-strength NLP processing
- **Hugging Face Transformers**: Modern transformer models
- **Gensim**: Topic modeling and document similarity
- **TextBlob**: Simple text processing
- **Stanford NLP**: Stanford's NLP library
- **AllenNLP**: Deep learning for NLP
- **Transformers**: Hugging Face transformers library
- **Scikit-learn**: Machine learning integration 