# spaCy Library

## Overview
spaCy is an industrial-strength Natural Language Processing (NLP) library for Python. It provides pre-trained models for multiple languages and offers fast, accurate linguistic analysis including tokenization, part-of-speech tagging, named entity recognition, dependency parsing, and more. spaCy is designed for production use and provides easy-to-use APIs for building NLP applications.

## Installation
```bash
# Basic installation
pip install spacy

# Download English language model
python -m spacy download en_core_web_sm

# Download larger English model
python -m spacy download en_core_web_md

# Download large English model with word vectors
python -m spacy download en_core_web_lg

# Install with GPU support
pip install spacy[cuda]

# Latest version
pip install spacy==3.7.2
```

## Key Features
- **Pre-trained Models**: High-quality models for multiple languages
- **Fast Processing**: Optimized for speed and efficiency
- **Production Ready**: Designed for real-world applications
- **Linguistic Features**: Tokenization, POS tagging, NER, parsing
- **Custom Training**: Easy fine-tuning and custom model training
- **Pipeline System**: Modular processing pipeline
- **Vector Representations**: Word vectors and embeddings
- **Rule-based Matching**: Pattern matching and rule extraction

## Core Concepts

### Basic Usage
```python
import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# Process text
text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

# Access tokens
for token in doc:
    print(f"{token.text:<15} {token.pos_:<10} {token.dep_:<10}")

# Access entities
for ent in doc.ents:
    print(f"{ent.text:<15} {ent.label_:<10}")

# Access noun chunks
for chunk in doc.noun_chunks:
    print(f"Noun chunk: {chunk.text}")
```

### Tokenization
```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Tokenization
text = "Apple Inc. is looking at buying U.K. startup for $1 billion"
doc = npc(text)

# Print tokens with their properties
for token in doc:
    print(f"Token: {token.text}")
    print(f"  Lemma: {token.lemma_}")
    print(f"  POS: {token.pos_}")
    print(f"  Tag: {token.tag_}")
    print(f"  Dep: {token.dep_}")
    print(f"  Shape: {token.shape_}")
    print(f"  Is alpha: {token.is_alpha}")
    print(f"  Is stop: {token.is_stop}")
    print()

# Sentence segmentation
text = "This is the first sentence. This is the second sentence. And this is the third."
doc = nlp(text)

for sent in doc.sents:
    print(f"Sentence: {sent.text}")
    print(f"Number of tokens: {len(sent)}")
    print()
```

## Linguistic Analysis

### Part-of-Speech Tagging
```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

# POS tagging
print("Part-of-Speech Analysis:")
for token in doc:
    print(f"{token.text:<15} {token.pos_:<10} {spacy.explain(token.pos_)}")

# Find specific POS tags
verbs = [token.text for token in doc if token.pos_ == "VERB"]
nouns = [token.text for token in doc if token.pos_ == "NOUN"]
adjectives = [token.text for token in doc if token.pos_ == "ADJ"]

print(f"\nVerbs: {verbs}")
print(f"Nouns: {nouns}")
print(f"Adjectives: {adjectives}")
```

### Named Entity Recognition (NER)
```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. CEO Tim Cook announced new products at the WWDC conference in San Francisco."
doc = nlp(text)

# Named entities
print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text:<20} {ent.label_:<15} {spacy.explain(ent.label_)}")

# Entity types
entity_types = {
    'PERSON': 'People, including fictional',
    'NORP': 'Nationalities, religious or political groups',
    'FAC': 'Buildings, airports, highways, bridges, etc.',
    'ORG': 'Companies, agencies, institutions, etc.',
    'GPE': 'Countries, cities, states',
    'LOC': 'Non-GPE locations, mountain ranges, bodies of water',
    'PRODUCT': 'Objects, vehicles, foods, etc.',
    'EVENT': 'Named hurricanes, battles, wars, sports events, etc.',
    'WORK_OF_ART': 'Titles of books, songs, etc.',
    'LAW': 'Named documents made into laws',
    'LANGUAGE': 'Any named language',
    'DATE': 'Absolute or relative dates or periods',
    'TIME': 'Times smaller than a day',
    'PERCENT': 'Percentage, including "%"',
    'MONEY': 'Monetary values, including unit',
    'QUANTITY': 'Measurements, as of weight or distance',
    'ORDINAL': '"first", "second", etc.',
    'CARDINAL': 'Numerals that do not fall under another type'
}

print("\nEntity Types:")
for entity_type, description in entity_types.items():
    print(f"{entity_type:<15} {description}")
```

### Dependency Parsing
```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "The cat sat on the mat."
doc = nlp(text)

# Dependency parsing
print("Dependency Parsing:")
for token in doc:
    print(f"{token.text:<10} {token.dep_:<15} {token.head.text:<10} {spacy.explain(token.dep_)}")

# Visualize dependency tree
from spacy import displacy

# Display dependency tree
displacy.serve(doc, style="dep")

# Find subject-verb-object relationships
def find_svo(doc):
    svo_triplets = []
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = token.text
            verb = token.head.text
            obj = None
            
            # Find object
            for child in token.head.children:
                if child.dep_ in ["dobj", "pobj"]:
                    obj = child.text
                    break
            
            if obj:
                svo_triplets.append((subject, verb, obj))
    
    return svo_triplets

svo_triplets = find_svo(doc)
print(f"\nSubject-Verb-Object relationships: {svo_triplets}")
```

## Text Processing

### Lemmatization and Stemming
```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "The cats are running and jumping over the fences."
doc = nlp(text)

print("Lemmatization:")
for token in doc:
    if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]:
        print(f"{token.text:<15} -> {token.lemma_}")

# Custom lemmatization
def custom_lemmatize(text):
    doc = nlp(text)
    lemmatized = []
    
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]:
            lemmatized.append(token.lemma_)
        else:
            lemmatized.append(token.text)
    
    return " ".join(lemmatized)

result = custom_lemmatize("The cats are running and jumping over the fences.")
print(f"\nCustom lemmatization: {result}")
```

### Stop Words Removal
```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

# Remove stop words
tokens_without_stops = [token.text for token in doc if not token.is_stop]
print(f"Original: {text}")
print(f"Without stop words: {' '.join(tokens_without_stops)}")

# Custom stop words
custom_stops = {"fox", "dog"}
nlp.Defaults.stop_words.update(custom_stops)

doc = nlp(text)
tokens_without_custom_stops = [token.text for token in doc if not token.is_stop]
print(f"Without custom stop words: {' '.join(tokens_without_custom_stops)}")
```

## Pattern Matching

### Rule-based Matching
```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Simple pattern matching
pattern = [{"LOWER": "hello"}, {"LOWER": "world"}]
matcher.add("HELLO_WORLD", [pattern])

text = "Hello world! Hello there!"
doc = nlp(text)
matches = matcher(doc)

for match_id, start, end in matches:
    span = doc[start:end]
    print(f"Match: {span.text}")

# Complex pattern matching
def find_phone_numbers(text):
    doc = nlp(text)
    
    # Pattern for phone numbers
    pattern = [
        {"ORTH": "("},
        {"SHAPE": "ddd"},
        {"ORTH": ")"},
        {"SHAPE": "ddd"},
        {"ORTH": "-"},
        {"SHAPE": "dddd"}
    ]
    
    matcher = Matcher(nlp.vocab)
    matcher.add("PHONE_NUMBER", [pattern])
    
    matches = matcher(doc)
    phone_numbers = []
    
    for match_id, start, end in matches:
        span = doc[start:end]
        phone_numbers.append(span.text)
    
    return phone_numbers

text = "Call me at (555) 123-4567 or (555) 987-6543"
phone_numbers = find_phone_numbers(text)
print(f"Phone numbers found: {phone_numbers}")
```

### Phrase Matching
```python
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)

# Add phrases to match
phrases = ["machine learning", "artificial intelligence", "deep learning"]
patterns = [nlp(phrase) for phrase in phrases]
matcher.add("AI_TERMS", patterns)

text = "Machine learning and artificial intelligence are transforming deep learning applications."
doc = nlp(text)
matches = matcher(doc)

for match_id, start, end in matches:
    span = doc[start:end]
    print(f"Found: {span.text}")
```

## Vector Representations

### Word Vectors
```python
import spacy
import numpy as np

# Load model with vectors
nlp = spacy.load("en_core_web_md")

# Get word vectors
text = "cat dog computer programming"
doc = nlp(text)

for token in doc:
    if token.has_vector:
        print(f"{token.text}: {token.vector[:5]}...")  # Show first 5 dimensions
        print(f"Vector norm: {token.vector_norm:.4f}")

# Similarity between words
word1 = nlp("cat")
word2 = nlp("dog")
word3 = nlp("computer")

print(f"Similarity between 'cat' and 'dog': {word1.similarity(word2):.4f}")
print(f"Similarity between 'cat' and 'computer': {word1.similarity(word3):.4f}")

# Document similarity
doc1 = nlp("I love cats")
doc2 = nlp("I love dogs")
doc3 = nlp("I love programming")

print(f"Similarity between doc1 and doc2: {doc1.similarity(doc2):.4f}")
print(f"Similarity between doc1 and doc3: {doc1.similarity(doc3):.4f}")
```

### Document Vectors
```python
import spacy

nlp = spacy.load("en_core_web_md")

# Document vectors
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand human language.",
    "Computer vision enables machines to interpret visual information."
]

docs = [nlp(doc) for doc in documents]

# Compare document similarities
for i, doc1 in enumerate(docs):
    for j, doc2 in enumerate(docs[i+1:], i+1):
        similarity = doc1.similarity(doc2)
        print(f"Doc {i+1} vs Doc {j+1}: {similarity:.4f}")
```

## Custom Pipeline Components

### Custom Pipeline Component
```python
import spacy
from spacy.language import Language

@Language.component("custom_component")
def custom_component(doc):
    # Add custom analysis
    for token in doc:
        # Add custom attribute
        token._.is_important = token.pos_ in ["NOUN", "VERB"] and not token.is_stop
    
    return doc

# Add component to pipeline
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("custom_component")

# Register custom attribute
from spacy.tokens import Token
Token.set_extension("is_important", default=False)

# Use custom component
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

for token in doc:
    if token._.is_important:
        print(f"Important token: {token.text}")
```

### Custom Entity Ruler
```python
import spacy
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_sm")

# Create entity ruler
ruler = EntityRuler(nlp)
patterns = [
    {"label": "TECHNOLOGY", "pattern": "machine learning"},
    {"label": "TECHNOLOGY", "pattern": "artificial intelligence"},
    {"label": "TECHNOLOGY", "pattern": "deep learning"},
    {"label": "COMPANY", "pattern": "Apple Inc."},
    {"label": "COMPANY", "pattern": "Google LLC"}
]

ruler.add_patterns(patterns)
nlp.add_pipe("entity_ruler")

# Test custom entities
text = "Apple Inc. is working on machine learning and artificial intelligence."
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text:<25} {ent.label_}")
```

## Model Training

### Custom NER Training
```python
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
import random

# Training data
TRAIN_DATA = [
    ("Apple Inc. is looking at buying U.K. startup for $1 billion", {
        "entities": [(0, 9, "ORG"), (27, 31, "GPE"), (44, 54, "MONEY")]
    }),
    ("Microsoft Corp. announced new products", {
        "entities": [(0, 15, "ORG")]
    })
]

def train_custom_ner():
    # Load blank model
    nlp = spacy.blank("en")
    
    # Add NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    # Add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    # Disable other pipes during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        
        for itn in range(10):
            random.shuffle(TRAIN_DATA)
            losses = {}
            
            for text, annotations in TRAIN_DATA:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, losses=losses)
            
            print(f"Iteration {itn}, Losses: {losses}")
    
    return nlp

# Train model
# trained_nlp = train_custom_ner()
```

## Use Cases
- **Text Analysis**: Linguistic analysis and text processing
- **Information Extraction**: Named entity recognition and relation extraction
- **Text Classification**: Document classification and sentiment analysis
- **Question Answering**: Building QA systems
- **Chatbots**: Natural language understanding for conversational AI
- **Content Analysis**: Analyzing large volumes of text
- **Research**: Academic and research applications in NLP
- **Production Systems**: Industrial-strength NLP applications

## Best Practices
1. **Choose Appropriate Models**: Select models based on your task and language
2. **Use Efficient Processing**: Process text in batches for large datasets
3. **Custom Components**: Add custom pipeline components for specific needs
4. **Model Training**: Fine-tune models on domain-specific data
5. **Memory Management**: Use appropriate model sizes for your use case
6. **Error Handling**: Handle cases where models may fail
7. **Performance Optimization**: Use GPU acceleration when available
8. **Regular Updates**: Keep models and spaCy updated

## Advantages
- **Production Ready**: Designed for real-world applications
- **Fast Processing**: Optimized for speed and efficiency
- **High Quality**: State-of-the-art linguistic analysis
- **Easy to Use**: Simple and intuitive API
- **Customizable**: Easy to extend and customize
- **Multiple Languages**: Support for many languages
- **Active Development**: Regular updates and improvements
- **Good Documentation**: Comprehensive documentation and tutorials

## Limitations
- **Model Size**: Large models require significant memory
- **Training Data**: Custom training requires annotated data
- **Language Support**: Not all languages have equally good models
- **Computational Cost**: Processing can be expensive for large texts
- **Domain Adaptation**: May need fine-tuning for specific domains

## Related Libraries
- **NLTK**: Natural language processing toolkit
- **Hugging Face Transformers**: Pre-trained transformer models
- **Gensim**: Topic modeling and document similarity
- **TextBlob**: Simple text processing
- **Stanford NLP**: Stanford's NLP library
- **AllenNLP**: Deep learning for NLP
- **Transformers**: Hugging Face transformers library 