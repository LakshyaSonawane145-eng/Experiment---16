# Experiment 16 — NLP Techniques on Text Data in Python

**Name:** Lakshya Sonawane &nbsp;|&nbsp; **PRN:** 25070123068

---

## Introduction

Natural Language Processing (NLP) is a branch of Artificial Intelligence that enables computers to understand, interpret, and process human language. Text data is inherently unstructured — it contains punctuation, grammatical variations, stopwords, and different word forms — making it difficult to use directly in analysis or machine learning models.

This experiment demonstrates the most fundamental NLP preprocessing techniques using Python's **NLTK (Natural Language Toolkit)** library. Each technique transforms raw text into a cleaner, more structured form that is suitable for downstream tasks like sentiment analysis, text classification, spam detection, and more.

---

## Requirements

```python
pip install nltk
```

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
```

---

## Theory & Commands

### 1. Tokenization

Tokenization is the process of splitting text into smaller units called **tokens** (words or sentences). It is the **first and most essential step** in any NLP pipeline — all subsequent techniques depend on it.

**Word Tokenization** — splits text into individual words:

```python
from nltk.tokenize import word_tokenize

text = "Natural language processing is interesting"
tokens = word_tokenize(text)
print(tokens)
# Output: ['Natural', 'language', 'processing', 'is', 'interesting']
```

**Sentence Tokenization** — splits a paragraph into individual sentences:

```python
from nltk.tokenize import sent_tokenize

text = "Python is easy. It is widely used in data science."
sentences = sent_tokenize(text)
print(sentences)
# Output: ['Python is easy.', 'It is widely used in data science.']
```

---

### 2. Stop Word Removal

**Stop words** are common words (such as *is, the, and, in, it*) that appear frequently in text but carry little meaningful information. Removing them reduces noise and improves the efficiency of NLP models.

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
tokens = word_tokenize(text)

filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(filtered_tokens)
# Output: ['Python', 'easy', '.', 'widely', 'used', 'data', 'science', '.']
```

> **Note:** Punctuation marks (`.`, `,`) are not stop words by default and may need separate handling using regex or `string.punctuation`.

---

### 3. Stemming

Stemming reduces words to their **root/base form** by chopping off suffixes. The result may not always be a valid dictionary word (e.g., `studies` → `studi`). It is a **rule-based, fast** approach.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["coding", "watched", "gone", "studies", "playing", "running", "runs", "ran"]

for w in words:
    print(stemmer.stem(w))
# Output: code, watch, gone, studi, play, run, run, ran
```

---

### 4. Lemmatization

Lemmatization also reduces words to their base form, but unlike stemming it uses a **vocabulary and morphological analysis** to return a valid dictionary word (called a **lemma**). It is slower but more accurate.

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ["coding", "watched", "gone", "studies", "playing", "running", "runs", "ran"]

for w in words:
    print(lemmatizer.lemmatize(w))
# Output: coding, watched, gone, study, playing, running, run, ran
```

**Stemming vs Lemmatization — Quick Comparison:**

| Feature | Stemming | Lemmatization |
|---|---|---|
| Method | Chops suffix (rule-based) | Uses dictionary lookup |
| Output | May not be a real word (`studi`) | Always a valid word (`study`) |
| Speed | Faster | Slower |
| Accuracy | Lower | Higher |
| Example | `running` → `run` | `running` → `running` (noun default) |

---

### 5. Part-of-Speech (POS) Tagging

POS tagging assigns a **grammatical label** to each word in a sentence, such as noun (NN), verb (VB), adjective (JJ), etc. It helps models understand the role of each word in context.

```python
from nltk import pos_tag
from nltk.tokenize import word_tokenize

words = word_tokenize("Python is a powerful programming language")
tags = pos_tag(words)
print(tags)
# Output: [('Python', 'NNP'), ('is', 'VBZ'), ('powerful', 'JJ'), ('programming', 'JJ'), ('language', 'NN')]
```

**Common POS Tags:**

| Tag | Meaning | Example |
|---|---|---|
| `NNP` | Proper noun (singular) | Python, India |
| `NN` | Common noun (singular) | language, dog |
| `VBZ` | Verb, 3rd person present | is, runs |
| `JJ` | Adjective | powerful, easy |
| `RB` | Adverb | widely, quickly |
| `IN` | Preposition | in, of, at |

---

### 6. Word Frequency Count

`FreqDist` (Frequency Distribution) counts how many times each word appears in the text. Useful for identifying the most significant words in a corpus.

```python
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

text = "Python is easy. It is widely used in data science."
words = word_tokenize(text)

fd = FreqDist(words)
print(fd.most_common())
# Output: [('is', 2), ('.', 2), ('Python', 1), ('easy', 1), ...]
```

---

## NLP Preprocessing Pipeline

```
Raw Text
   │
   ▼
Tokenization          →  Split into words/sentences
   │
   ▼
Lowercasing           →  Normalize case
   │
   ▼
Stop Word Removal     →  Remove common, low-value words
   │
   ▼
Punctuation Removal   →  Clean symbols
   │
   ▼
Stemming / Lemmatization  →  Reduce to root/base form
   │
   ▼
POS Tagging           →  Tag grammatical roles
   │
   ▼
Clean, Structured Text  →  Ready for ML / Analysis
```

---

## Conclusion

This experiment covered the core NLP preprocessing techniques using Python's NLTK library:

- **Tokenization** splits raw text into words or sentences, forming the foundation of all NLP tasks.
- **Stop Word Removal** eliminates common, low-information words, reducing dataset size and noise.
- **Stemming** quickly reduces words to their root form using suffix-stripping rules, though it may produce non-dictionary words.
- **Lemmatization** produces accurate base forms using vocabulary lookup — preferred when precision matters.
- **POS Tagging** annotates each token with its grammatical role, enabling deeper language understanding.
- **Word Frequency Count** using `FreqDist` identifies the most common and important terms in a text.

Together, these techniques form a complete text preprocessing pipeline that converts unstructured raw text into clean, structured data ready for machine learning models, sentiment analysis, chatbots, search engines, and other NLP applications.

---

## Libraries Used

| Library | Purpose |
|---|---|
| `nltk` | Core NLP operations — tokenization, stemming, POS tagging |
| `nltk.tokenize` | `word_tokenize`, `sent_tokenize` |
| `nltk.corpus` | Stopwords list |
| `nltk.stem` | `PorterStemmer`, `WordNetLemmatizer` |
| `nltk.probability` | `FreqDist` for word frequency |
