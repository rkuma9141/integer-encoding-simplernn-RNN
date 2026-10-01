# Integer Encoding in RNN

## 📌 Overview

This project demonstrates how **Integer Encoding** is used to convert text words into numerical values before feeding the text into an **RNN (Recurrent Neural Network)**.

Machine Learning and Deep Learning models cannot directly understand text. Therefore, words are first converted into integers, and these integer sequences are then used as input to the RNN.

---

## 🔹 What is Integer Encoding?

**Integer Encoding** is a text preprocessing technique where each unique word is assigned a unique integer.

### Example

Suppose we have the sentences:

```text
I love machine learning
I love deep learning
```

After creating a vocabulary, we may get:

```text
I        → 1
love     → 2
machine  → 3
learning → 4
deep     → 5
```

The sentences become:

```text
I love machine learning
↓
1  2  3  4

I love deep learning
↓
1  2  5  4
```

These integer sequences can then be provided to an RNN.

---

## 🔹 Why Do We Need Integer Encoding?

Neural networks work with numerical data, not raw text.

For example:

```text
"I love AI"
```

cannot directly be given to an RNN.

After integer encoding:

```text
[1, 2, 3]
```

the sequence can be processed by the neural network.

---

## 🔹 Integer Encoding vs One-Hot Encoding

### Integer Encoding

Each word gets one integer:

```text
I      → 1
love   → 2
AI     → 3
```

Advantages:

* Requires less memory
* Simple representation
* Commonly used before an Embedding layer

### One-Hot Encoding

Each word is represented as a vector.

For example:

```text
I     → [1, 0, 0]
love  → [0, 1, 0]
AI    → [0, 0, 1]
```

One-hot vectors can become very large when the vocabulary is large.

---

## 🔹 Integer Encoding with TensorFlow

```python
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    "I love machine learning",
    "I love deep learning"
]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

print(word_index)
```

Example output:

```text
{'i': 1, 'love': 2, 'learning': 3, 'machine': 4, 'deep': 5}
```

---

## 🔹 Convert Sentences into Integer Sequences

```python
sequences = tokenizer.texts_to_sequences(sentences)

print(sequences)
```

Output:

```text
[[1, 2, 4, 3],
 [1, 2, 5, 3]]
```

Here:

```text
"I love machine learning"
        ↓
[1, 2, 4, 3]
```

and:

```text
"I love deep learning"
        ↓
[1, 2, 5, 3]
```

---

## 🔹 Padding Sequences

RNN models generally require inputs to have the same sequence length.

For example:

```text
[1, 2, 4]
[1, 2, 5, 3]
```

have different lengths.

We can use padding:

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_sequences = pad_sequences(
    sequences,
    padding='post'
)

print(padded_sequences)
```

Output:

```text
[
 [1, 2, 4, 3],
 [1, 2, 5, 3]
]
```

For different-length sequences, padding adds zeros.

Example:

```text
[1, 2, 4]
[1, 2, 5, 3]

↓ padding

[1, 2, 4, 0]
[1, 2, 5, 3]
```

---

## 🔹 Integer Encoding → Embedding → RNN

A typical text-processing pipeline is:

```text
Raw Text
   ↓
Tokenization
   ↓
Integer Encoding
   ↓
Padding
   ↓
Embedding Layer
   ↓
RNN / LSTM / GRU
   ↓
Dense Layer
   ↓
Prediction
```

### Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

model = Sequential([
    Embedding(input_dim=10000, output_dim=64),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.summary()
```

---

## 🔹 Why is an Embedding Layer Used?

Integer values themselves do **not** represent the meaning of words.

For example:

```text
cat → 1
dog → 2
car → 3
```

The number `2` does not mean that "dog" is somehow twice "cat".

The **Embedding layer** converts these integer IDs into dense vectors that the neural network can learn from.

Example:

```text
cat → 1 → [0.21, 0.45, 0.18, ...]
dog → 2 → [0.19, 0.42, 0.22, ...]
car → 3 → [0.81, 0.12, 0.67, ...]
```

---

## 🔹 Complete Example

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

sentences = [
    "I love machine learning",
    "I love deep learning",
    "RNN is useful for text",
    "Deep learning is powerful"
]

# 1. Create tokenizer
tokenizer = Tokenizer()

# 2. Build vocabulary
tokenizer.fit_on_texts(sentences)

# 3. Integer encoding
sequences = tokenizer.texts_to_sequences(sentences)

print("Word Index:")
print(tokenizer.word_index)

print("\nInteger Sequences:")
print(sequences)

# 4. Padding
X = pad_sequences(
    sequences,
    padding='post'
)

print("\nPadded Sequences:")
print(X)

# 5. Build RNN model
vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=16
    ),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.summary()
```

---

## 🔹 Important Terms

| Term             | Meaning                                          |
| ---------------- | ------------------------------------------------ |
| Tokenization     | Breaking text into smaller units                 |
| Vocabulary       | Collection of unique words                       |
| Integer Encoding | Assigning an integer ID to each word             |
| Padding          | Making sequences the same length                 |
| Embedding        | Converting word IDs into dense vectors           |
| RNN              | Neural network designed for sequential data      |
| LSTM             | Improved RNN that handles long-term dependencies |
| GRU              | Lightweight alternative to LSTM                  |

---

## 🔹 Real-World Applications

Integer encoding and RNN-based models can be used for:

* Sentiment Analysis
* Text Classification
* Spam Detection
* Language Modeling
* Text Generation
* Chatbots
* Sequence Prediction

---

## 🎯 Interview Answer

### What is Integer Encoding?

**Answer:**

> Integer encoding is a text preprocessing technique in which each unique word is assigned a unique integer ID. The resulting integer sequences are usually padded and passed through an embedding layer before being processed by an RNN, LSTM, or GRU.

### Example:

```text
"I love AI"

I    → 1
love → 2
AI   → 3

Result:
[1, 2, 3]
```

The important point is that **integer encoding only represents words using IDs; the Embedding layer learns meaningful numerical representations of those IDs.**
