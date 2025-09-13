# Hugging Face Transformers Library

## Overview
Hugging Face Transformers is a comprehensive library that provides thousands of pre-trained models for Natural Language Processing (NLP) tasks. It offers state-of-the-art models like BERT, GPT, T5, RoBERTa, and many others, along with easy-to-use APIs for text classification, translation, summarization, question answering, and more. The library is built on PyTorch and TensorFlow, making it accessible to a wide range of users.

## Installation
```bash
# Basic installation
pip install transformers

# With PyTorch
pip install transformers[torch]

# With TensorFlow
pip install transformers[tf]

# With all dependencies
pip install transformers[all]

# Latest version
pip install transformers==4.35.0
```

## Key Features
- **Pre-trained Models**: Access to thousands of pre-trained models
- **Multiple Frameworks**: Support for PyTorch and TensorFlow
- **Easy-to-use APIs**: Simple interfaces for common NLP tasks
- **Model Hub**: Access to community-contributed models
- **Tokenization**: Fast and efficient tokenization
- **Pipeline API**: High-level API for common tasks
- **Custom Training**: Fine-tune models on your own data
- **Model Sharing**: Share and deploy models easily

## Core Concepts

### Basic Usage with Pipeline
```python
from transformers import pipeline

# Text classification
classifier = pipeline("text-classification")
result = classifier("I love this movie!")
print(result)

# Sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis")
result = sentiment_analyzer("I am very happy today!")
print(result)

# Text generation
generator = pipeline("text-generation")
result = generator("The future of AI is")
print(result[0]['generated_text'])

# Question answering
qa_pipeline = pipeline("question-answering")
context = "Hugging Face is a company that develops tools for building machine learning applications."
question = "What does Hugging Face do?"
result = qa_pipeline(question=question, context=context)
print(result)
```

### Using Pre-trained Models
```python
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize text
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs)

print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Output shape: {outputs.last_hidden_state.shape}")
```

## Text Classification

### Sequence Classification
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model for classification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare text
texts = ["I love this movie!", "This is terrible.", "It's okay."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get labels
labels = model.config.id2label
for i, text in enumerate(texts):
    pred_label = labels[predictions[i].argmax().item()]
    confidence = predictions[i].max().item()
    print(f"Text: {text}")
    print(f"Prediction: {pred_label} (confidence: {confidence:.3f})")
    print()
```

### Custom Classification
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Prepare data
texts = ["I love this", "I hate this", "This is neutral"] * 10
labels = [0, 1, 2] * 10  # 0: positive, 1: negative, 2: neutral

# Create dataset
dataset = Dataset.from_dict({
    'text': texts,
    'label': labels
})

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train model
trainer.train()
```

## Text Generation

### Language Generation
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model for text generation
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

# Generate text
prompt = "The future of artificial intelligence"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=3,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode outputs
for i, output in enumerate(outputs):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Generated text {i+1}: {generated_text}")
    print()
```

### Translation
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load translation model
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Translate text
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

# Generate translation
with torch.no_grad():
    outputs = model.generate(**inputs)

translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"English: {text}")
print(f"French: {translation}")
```

## Question Answering

### Extractive QA
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load QA model
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Prepare context and question
context = """
Hugging Face is a company that develops tools for building machine learning applications. 
The company was founded in 2016 and is based in New York City. 
They are known for their popular transformers library and model hub.
"""

question = "When was Hugging Face founded?"

# Tokenize
inputs = tokenizer(
    question,
    context,
    return_tensors="pt",
    max_length=512,
    truncation=True
)

# Get answer
with torch.no_grad():
    outputs = model(**inputs)

# Process outputs
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1

answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0][answer_start:answer_end]
    )
)

print(f"Question: {question}")
print(f"Answer: {answer}")
```

## Named Entity Recognition (NER)

### Entity Recognition
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load NER model
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Prepare text
text = "Hugging Face is a company based in New York City."

# Tokenize
inputs = tokenizer(text, return_tensors="pt", truncation=True)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)

# Process predictions
predictions = torch.argmax(outputs.logits, dim=2)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Map predictions to labels
id2label = model.config.id2label
entities = []
current_entity = None

for token, prediction in zip(tokens, predictions[0]):
    if token.startswith("##"):
        continue
    
    label = id2label[prediction.item()]
    
    if label.startswith("B-"):
        if current_entity:
            entities.append(current_entity)
        current_entity = {"text": token, "label": label[2:]}
    elif label.startswith("I-") and current_entity and label[2:] == current_entity["label"]:
        current_entity["text"] += " " + token
    else:
        if current_entity:
            entities.append(current_entity)
        current_entity = None

if current_entity:
    entities.append(current_entity)

print("Named Entities:")
for entity in entities:
    print(f"{entity['text']}: {entity['label']}")
```

## Model Fine-tuning

### Custom Fine-tuning
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prepare custom data
texts = [
    "This product is amazing!",
    "I love this service",
    "Terrible experience",
    "Not worth the money",
    "Excellent quality",
    "Poor customer service"
] * 50

labels = [1, 1, 0, 0, 1, 0] * 50  # 1: positive, 0: negative

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Create datasets
train_dataset = Dataset.from_dict({
    'text': train_texts,
    'label': train_labels
})

val_dataset = Dataset.from_dict({
    'text': val_texts,
    'label': val_labels
})

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    load_best_model_at_end=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save model
trainer.save_model("./sentiment_model_final")
tokenizer.save_pretrained("./sentiment_model_final")
```

## Model Sharing and Deployment

### Save and Load Models
```python
from transformers import AutoTokenizer, AutoModel

# Save model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save locally
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Load from local
loaded_tokenizer = AutoTokenizer.from_pretrained("./my_model")
loaded_model = AutoModel.from_pretrained("./my_model")

# Push to Hub (requires authentication)
# model.push_to_hub("my-username/my-model")
# tokenizer.push_to_hub("my-username/my-model")
```

### Model Pipelines
```python
from transformers import pipeline

# Create custom pipeline
def create_custom_pipeline():
    # Load model and tokenizer
    model_name = "./sentiment_model_final"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Create pipeline
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    return classifier

# Use custom pipeline
custom_classifier = create_custom_pipeline()
result = custom_classifier("This is a great product!")
print(result)
```

## Use Cases
- **Text Classification**: Sentiment analysis, topic classification, intent detection
- **Text Generation**: Language modeling, story generation, code generation
- **Translation**: Machine translation between languages
- **Question Answering**: Extractive and generative QA systems
- **Named Entity Recognition**: Entity extraction from text
- **Summarization**: Text summarization and compression
- **Conversational AI**: Chatbots and dialogue systems
- **Code Generation**: Programming language generation

## Best Practices
1. **Choose Appropriate Models**: Select models based on your task and data
2. **Use Pre-trained Models**: Leverage pre-trained models when possible
3. **Proper Tokenization**: Ensure correct tokenization for your model
4. **Batch Processing**: Use batch processing for efficiency
5. **Model Caching**: Cache models to avoid repeated downloads
6. **Memory Management**: Use appropriate batch sizes and model sizes
7. **Evaluation**: Use proper evaluation metrics for your task
8. **Model Sharing**: Share models through Hugging Face Hub

## Advantages
- **Pre-trained Models**: Access to thousands of state-of-the-art models
- **Easy Integration**: Simple APIs for common NLP tasks
- **Multiple Frameworks**: Support for PyTorch and TensorFlow
- **Active Community**: Large community and regular updates
- **Model Hub**: Easy sharing and discovery of models
- **Comprehensive**: Covers most NLP tasks and use cases
- **Production Ready**: Suitable for production deployments

## Limitations
- **Model Size**: Large models require significant computational resources
- **Fine-tuning Cost**: Training can be expensive and time-consuming
- **Domain Adaptation**: May need fine-tuning for specific domains
- **Interpretability**: Complex models are less interpretable
- **Bias**: Models may inherit biases from training data

## Related Libraries
- **PyTorch**: Deep learning framework
- **TensorFlow**: Alternative deep learning framework
- **Datasets**: Data loading and processing
- **Tokenizers**: Fast tokenization library
- **Accelerate**: Distributed training utilities
- **Optimum**: Model optimization library
- **Gradio**: Model deployment and UI 