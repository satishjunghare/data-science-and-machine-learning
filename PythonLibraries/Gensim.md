# Gensim

## Overview

**Gensim** is a Python library for topic modeling and document similarity analysis. It's designed to handle large text collections efficiently and provides implementations of popular algorithms like Latent Semantic Analysis (LSA), Latent Dirichlet Allocation (LDA), and Word2Vec. Gensim is particularly useful for natural language processing tasks involving document similarity, topic discovery, and word embeddings.

## Installation

```bash
# Install Gensim
pip install gensim

# Install with additional features
pip install gensim[fasttext]  # For FastText support
pip install gensim[levenshtein]  # For Levenshtein distance
pip install gensim[mallet]  # For Mallet LDA

# Install with all optional dependencies
pip install gensim[all]
```

## Key Features

- **Topic modeling**: LDA, LSA, HDP, and other algorithms
- **Word embeddings**: Word2Vec, Doc2Vec, FastText
- **Document similarity**: Cosine similarity, semantic similarity
- **Large-scale processing**: Memory-efficient processing of large corpora
- **Multiple formats**: Support for various input/output formats
- **Pre-trained models**: Access to pre-trained word vectors
- **Streaming**: Online learning and incremental updates
- **Visualization**: Built-in visualization tools

## Core Concepts

### 1. Document and Corpus
Documents are represented as bags of words, and a corpus is a collection of documents.

```python
from gensim import corpora, models
from gensim.utils import simple_preprocess

# Sample documents
documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement"
]

# Preprocess documents
def preprocess(docs):
    return [simple_preprocess(doc, deacc=True) for doc in docs]

processed_docs = preprocess(documents)

# Create dictionary
dictionary = corpora.Dictionary(processed_docs)

# Create corpus (bag of words)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

print("Dictionary:", dictionary.token2id)
print("Corpus:", corpus)
```

### 2. Topic Models
Topic models discover abstract topics in a collection of documents.

```python
# Create LDA model
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,
    random_state=42,
    passes=15,
    alpha='auto',
    per_word_topics=True
)

# Print topics
for idx, topic in lda_model.print_topics(-1):
    print(f'Topic: {idx} \nWords: {topic}')
```

### 3. Word Embeddings
Word embeddings represent words as dense vectors in a continuous vector space.

```python
from gensim.models import Word2Vec

# Train Word2Vec model
w2v_model = Word2Vec(
    sentences=processed_docs,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# Get word vector
vector = w2v_model.wv['computer']
print("Vector for 'computer':", vector[:10])  # First 10 dimensions
```

## Basic Usage

### Document Preprocessing
```python
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

# Download required NLTK data
nltk.download('wordnet')
nltk.download('punkt')

# Initialize stemmer and lemmatizer
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def lemmatize_stemming(text):
    return stemmer.stem(lemmatizer.lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# Preprocess documents
processed_docs = [preprocess(doc) for doc in documents]
print("Processed documents:", processed_docs)
```

### Creating Dictionary and Corpus
```python
from gensim import corpora

# Create dictionary
dictionary = corpora.Dictionary(processed_docs)

# Filter extreme frequencies
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# Create bag of words corpus
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Create TF-IDF corpus
tfidf = models.TfidfModel(bow_corpus)
tfidf_corpus = tfidf[bow_corpus]

print("Dictionary size:", len(dictionary))
print("Number of documents:", len(bow_corpus))
```

## Topic Modeling

### Latent Dirichlet Allocation (LDA)
```python
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# Create LDA model
lda_model = LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    num_topics=5,
    random_state=42,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

# Print topics
for idx, topic in lda_model.print_topics(-1):
    print(f'Topic {idx}: {topic}')

# Get topic distribution for a document
doc_topics = lda_model.get_document_topics(bow_corpus[0])
print("Document topics:", doc_topics)

# Compute model coherence
coherence_model_lda = CoherenceModel(
    model=lda_model, 
    texts=processed_docs, 
    dictionary=dictionary, 
    coherence='c_v'
)
coherence_lda = coherence_model_lda.get_coherence()
print(f'Coherence Score: {coherence_lda}')
```

### Finding Optimal Number of Topics
```python
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit, step):
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        model_list.append(model)
        
        coherencemodel = CoherenceModel(
            model=model, 
            texts=texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_values.append(coherencemodel.get_coherence())
    
    return model_list, coherence_values

# Compute coherence for different numbers of topics
model_list, coherence_values = compute_coherence_values(
    dictionary=dictionary, 
    corpus=bow_corpus, 
    texts=processed_docs, 
    start=2, 
    limit=40, 
    step=6
)

# Plot results
import matplotlib.pyplot as plt

x = range(2, 40, 6)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
```

### Hierarchical Dirichlet Process (HDP)
```python
from gensim.models import HdpModel

# Create HDP model
hdp_model = HdpModel(
    corpus=bow_corpus,
    id2word=dictionary
)

# Print topics
for idx, topic in hdp_model.print_topics(-1):
    print(f'Topic {idx}: {topic}')

# Get number of topics
num_topics = len(hdp_model.get_topics())
print(f"Number of topics: {num_topics}")
```

## Word Embeddings

### Word2Vec
```python
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Train Word2Vec model
w2v_model = Word2Vec(
    sentences=processed_docs,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=0,  # CBOW
    epochs=10
)

# Find similar words
similar_words = w2v_model.wv.most_similar('computer')
print("Words similar to 'computer':", similar_words)

# Find word similarity
similarity = w2v_model.wv.similarity('computer', 'system')
print(f"Similarity between 'computer' and 'system': {similarity}")

# Word analogy
result = w2v_model.wv.most_similar(positive=['computer', 'human'], negative=['interface'])
print("computer - interface + human =", result[0][0])

# Save and load model
w2v_model.save("word2vec.model")
loaded_model = Word2Vec.load("word2vec.model")
```

### Doc2Vec
```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Create tagged documents
tagged_data = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(processed_docs)]

# Train Doc2Vec model
doc2vec_model = Doc2Vec(
    tagged_data,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=20
)

# Get document vector
doc_vector = doc2vec_model.dv[0]
print("Document vector:", doc_vector[:10])

# Find similar documents
similar_docs = doc2vec_model.dv.most_similar(0)
print("Documents similar to document 0:", similar_docs)

# Infer vector for new document
new_doc = preprocess("New document about computer systems")
new_vector = doc2vec_model.infer_vector(new_doc)
print("New document vector:", new_vector[:10])
```

### FastText
```python
from gensim.models import FastText

# Train FastText model
fasttext_model = FastText(
    sentences=processed_docs,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,  # Skip-gram
    epochs=10
)

# Handle out-of-vocabulary words
oov_vector = fasttext_model.wv['unknownword']
print("Vector for unknown word:", oov_vector[:10])

# Find similar words
similar_words = fasttext_model.wv.most_similar('computer')
print("Words similar to 'computer':", similar_words)
```

## Document Similarity

### Cosine Similarity
```python
from gensim.similarities import SparseMatrixSimilarity
from gensim.similarities import MatrixSimilarity
import numpy as np

# Create similarity index
index = SparseMatrixSimilarity(tfidf_corpus, num_features=len(dictionary))

# Get similarity scores for a query
query_doc = preprocess("computer system interface")
query_bow = dictionary.doc2bow(query_doc)
query_tfidf = tfidf[query_bow]

# Get similarity scores
sims = index[query_tfidf]
print("Similarity scores:", sims)

# Get most similar documents
similar_docs = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
print("Most similar documents:", similar_docs)
```

### Semantic Similarity
```python
from gensim.similarities import WmdSimilarity
from gensim.models import Word2Vec

# Train Word2Vec model
w2v_model = Word2Vec(processed_docs, vector_size=100, window=5, min_count=1, workers=4)

# Create WMD similarity
wmd_similarity = WmdSimilarity(processed_docs, w2v_model, num_best=10)

# Query
query = preprocess("computer system interface")

# Get similar documents
similar_docs = wmd_similarity[query]
print("Semantically similar documents:", similar_docs)
```

## Advanced Features

### Mallet LDA
```python
from gensim.models.wrappers import LdaMallet

# Train Mallet LDA (requires Mallet installation)
mallet_path = '/path/to/mallet-2.0.8/bin/mallet'
ldamallet = LdaMallet(
    mallet_path, 
    corpus=bow_corpus, 
    num_topics=5, 
    id2word=dictionary
)

# Print topics
for idx, topic in ldamallet.print_topics(-1):
    print(f'Topic {idx}: {topic}')
```

### Online Learning
```python
# Create model for online learning
lda_model = LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    num_topics=5,
    random_state=42,
    passes=1
)

# Update model with new documents
new_docs = [
    "New document about machine learning",
    "Another document about artificial intelligence"
]

new_processed_docs = [preprocess(doc) for doc in new_docs]
new_corpus = [dictionary.doc2bow(doc) for doc in new_processed_docs]

# Update model
lda_model.update(new_corpus)
```

### Model Persistence
```python
# Save models
lda_model.save('lda_model.model')
w2v_model.save('word2vec_model.model')
dictionary.save('dictionary.dict')

# Load models
loaded_lda = LdaModel.load('lda_model.model')
loaded_w2v = Word2Vec.load('word2vec_model.model')
loaded_dict = corpora.Dictionary.load('dictionary.dict')
```

## Visualization

### Topic Visualization
```python
import pyLDAvis
import pyLDAvis.gensim_models

# Prepare visualization
lda_visualization = pyLDAvis.gensim_models.prepare(
    lda_model, 
    bow_corpus, 
    dictionary
)

# Display visualization
pyLDAvis.show(lda_visualization)
```

### Word Embeddings Visualization
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Get word vectors
words = list(w2v_model.wv.key_to_index.keys())
word_vectors = [w2v_model.wv[word] for word in words]

# Reduce dimensionality
tsne = TSNE(n_components=2, random_state=42)
word_vectors_2d = tsne.fit_transform(word_vectors)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]))
plt.show()
```

## Use Cases

### 1. Document Classification
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Prepare data
documents = [
    "Machine learning algorithms for data analysis",
    "Computer vision applications in robotics",
    "Natural language processing techniques",
    "Deep learning for image recognition",
    "Statistical methods in data science"
]

labels = ['ml', 'cv', 'nlp', 'dl', 'stats']

# Create TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Train classifier
classifier = MultinomialNB()
classifier.fit(X, labels)

# Classify new document
new_doc = "Neural networks for pattern recognition"
new_doc_vector = vectorizer.transform([new_doc])
prediction = classifier.predict(new_doc_vector)
print(f"Predicted class: {prediction[0]}")
```

### 2. Topic Discovery
```python
# Large document collection
large_documents = [
    "Document about machine learning and artificial intelligence",
    "Text about computer vision and image processing",
    "Article on natural language processing and text analysis",
    "Paper about deep learning and neural networks",
    "Research on data science and statistical analysis"
]

# Preprocess
processed_large_docs = [preprocess(doc) for doc in large_documents]
large_dictionary = corpora.Dictionary(processed_large_docs)
large_corpus = [large_dictionary.doc2bow(doc) for doc in processed_large_docs]

# Train LDA
lda_large = LdaModel(
    corpus=large_corpus,
    id2word=large_dictionary,
    num_topics=3,
    random_state=42,
    passes=15
)

# Discover topics
for idx, topic in lda_large.print_topics(-1):
    print(f'Topic {idx}: {topic}')
```

### 3. Semantic Search
```python
# Create semantic search index
from gensim.similarities import SparseMatrixSimilarity

# Train Word2Vec
w2v_search = Word2Vec(processed_docs, vector_size=100, window=5, min_count=1)

# Create TF-IDF index
tfidf_search = models.TfidfModel(bow_corpus)
tfidf_corpus_search = tfidf_search[bow_corpus]
index_search = SparseMatrixSimilarity(tfidf_corpus_search, num_features=len(dictionary))

# Search function
def semantic_search(query, documents, index, dictionary, tfidf_model, top_k=5):
    # Preprocess query
    query_processed = preprocess(query)
    query_bow = dictionary.doc2bow(query_processed)
    query_tfidf = tfidf_model[query_bow]
    
    # Get similarities
    sims = index[query_tfidf]
    
    # Get top results
    top_results = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [(documents[i], score) for i, score in top_results]

# Search
query = "computer interface"
results = semantic_search(query, documents, index_search, dictionary, tfidf_search)
print("Search results:", results)
```

## Best Practices

### 1. Data Preprocessing
- Remove stop words and punctuation
- Apply lemmatization or stemming
- Handle case sensitivity
- Remove rare and frequent words

### 2. Model Selection
- Choose appropriate topic model (LDA, HDP, etc.)
- Determine optimal number of topics
- Use coherence scores for evaluation
- Consider computational requirements

### 3. Parameter Tuning
- Tune hyperparameters for better performance
- Use cross-validation when possible
- Monitor model convergence
- Regularize to prevent overfitting

### 4. Evaluation
- Use multiple evaluation metrics
- Validate results with domain experts
- Test on held-out data
- Consider interpretability

## Advantages

1. **Efficient**: Memory-efficient processing of large corpora
2. **Scalable**: Handles large-scale text collections
3. **Flexible**: Multiple algorithms and approaches
4. **Well-documented**: Comprehensive documentation and examples
5. **Active development**: Regular updates and improvements
6. **Integration**: Easy integration with other NLP tools
7. **Pre-trained models**: Access to pre-trained word vectors
8. **Streaming**: Support for online learning

## Limitations

1. **Learning curve**: Requires understanding of NLP concepts
2. **Parameter tuning**: Many parameters to tune
3. **Interpretability**: Topic models can be hard to interpret
4. **Quality dependence**: Results depend on data quality
5. **Computational cost**: Can be expensive for large datasets
6. **Black-box nature**: Limited interpretability of embeddings

## Related Libraries

- **NLTK**: Natural language processing toolkit
- **spaCy**: Industrial-strength NLP library
- **scikit-learn**: Machine learning library
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **pyLDAvis**: Topic model visualization
- **Word2Vec**: Word embeddings (Google) 