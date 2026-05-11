<div align="center">

# 🎬 CINESCOPE

### *Because every review tells a story.*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/▶%20WATCH%20LIVE-FF0000?style=for-the-badge)](https://cinescope-rnn.streamlit.app)

---

> **"Is this review worth your time — or a total flop?"**
> CineScope uses Deep Learning to find out.

</div>

---

## 🍿 WHAT IS CINESCOPE?

CineScope is an end-to-end **NLP Sentiment Analysis** web application that reads IMDB movie reviews and instantly classifies them as **Positive 👍** or **Negative 👎** — powered by a Simple Recurrent Neural Network (RNN) trained on 50,000 real movie reviews.

Type any review. Get the verdict. In seconds.

---

## 🎥 THE CAST  *(Tech Stack)*

| Role | Technology |
|------|-----------|
| 🧠 Brain | Simple RNN (128 neurons) |
| 📚 Language | Embedding Layer (10,000 vocab × 128 dims) |
| 🎬 Director | TensorFlow / Keras |
| 🖥️ Screen | Streamlit |
| 🗄️ Dataset | IMDB (50,000 reviews) |
| 🐍 Script | Python 3.8+ |

---

## 🎞️ THE PLOT  *(How It Works)*

```
Your Review Text
      │
      ▼
 ┌─────────────────────────────────┐
 │   TOKENISATION & CLEANING       │  → lowercase, strip punctuation
 └─────────────────────────────────┘
      │
      ▼
 ┌─────────────────────────────────┐
 │   INTEGER ENCODING              │  → "great" → 85, "movie" → 17 ...
 └─────────────────────────────────┘
      │
      ▼
 ┌─────────────────────────────────┐
 │   PADDING  (maxlen = 500)       │  → [0, 0, 0, 85, 17, ...]
 └─────────────────────────────────┘
      │
      ▼
 ┌─────────────────────────────────┐
 │   EMBEDDING LAYER               │  → 128-dim dense vectors
 │   10,000 × 128 = 1,280,000 params│
 └─────────────────────────────────┘
      │
      ▼
 ┌─────────────────────────────────┐
 │   SIMPLE RNN  (128 neurons)     │  → h_t = relu(Wx·x_t + Wh·h_{t-1} + b)
 │   32,896 params                 │
 └─────────────────────────────────┘
      │
      ▼
 ┌─────────────────────────────────┐
 │   DENSE (sigmoid)               │  → score between 0 and 1
 │   129 params                    │
 └─────────────────────────────────┘
      │
      ▼
  0.85 → 😍 POSITIVE
  0.23 → 💀 NEGATIVE
```

**Total Parameters: 1,313,025**

---

## 🏆 MODEL SPECS

```
Model: Sequential
─────────────────────────────────────────────────
Layer               Output Shape        Params
─────────────────────────────────────────────────
Embedding           (None, 500, 128)    1,280,000
SimpleRNN           (None, 128)            32,896
Dense (sigmoid)     (None, 1)                 129
─────────────────────────────────────────────────
Total params:                           1,313,025
Trainable params:                       1,313,025
─────────────────────────────────────────────────
```

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Epochs | Up to 7 (EarlyStopping) |
| Batch Size | 32 |
| Validation Split | 20% |
| Patience | 3 |

---

## 📺 EPISODES  *(Project Structure)*

```
CineScope/
│
├── 📓 simpleRNN.ipynb          # Model training — where the magic happens
├── 📓 RNNprediction.ipynb      # Prediction notebook — test any review
├── 🤖 imdb_rnn_model.h5        # Saved trained model
├── 🌐 app.py                   # Streamlit web app
└── 📄 README.md                # You are here
```

---

## 🎬 WATCH NOW  *(Run Locally)*

**Step 1 — Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/CineScope.git
cd CineScope
```

**Step 2 — Set up your environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install tensorflow keras streamlit numpy
```

**Step 3 — Press Play**
```bash
streamlit run app.py
```

**Step 4 — Go to `http://localhost:8501` and enjoy 🍿**

---

## 🌐 STREAM ONLINE

> No installation needed. Just click and go.

**👉 [Launch CineScope Live](https://cinescope-rnn.streamlit.app)**

---

## 📊 TRAINING PERFORMANCE

```
Epoch 1/7  →  loss: 0.6821  accuracy: 0.5612  val_loss: 0.6234
Epoch 2/7  →  loss: 0.5341  accuracy: 0.7234  val_loss: 0.4912
Epoch 3/7  →  loss: 0.3987  accuracy: 0.8301  val_loss: 0.3876
Epoch 4/7  →  loss: 0.3124  accuracy: 0.8720  val_loss: 0.3542
...
EarlyStopping triggered → Best weights restored ✅
```

---

## 🎭 EXAMPLE REVIEWS

| Review | Verdict | Score |
|--------|---------|-------|
| *"Absolutely brilliant film. A masterpiece."* | 😍 Positive | 0.91 |
| *"Worst movie I have ever seen. Complete waste."* | 💀 Negative | 0.08 |
| *"The acting was decent but the plot was boring."* | 💀 Negative | 0.32 |
| *"A stunning visual experience with a great story!"* | 😍 Positive | 0.87 |

---

## 🧠 THE SCIENCE BEHIND IT

### Why RNN?
Reviews are **sequential** — the order of words matters. "The movie was not good" is very different from "The movie was good." RNNs read text word by word and carry memory forward through time using a hidden state:

```
h_t = relu( Wx · x_t  +  Wh · h_{t-1}  +  b )
```

### Why Embeddings?
Raw integers like `[85, 17, 203]` carry no meaning. The Embedding layer converts each word into a **128-dimensional dense vector** — capturing semantic relationships learned during training.

### Why Binary Cross-Entropy?
This is a 2-class problem (positive or negative). BCE measures how far our sigmoid output is from the true label (0 or 1) and drives the model to improve.

---

## 👨‍💻 DIRECTOR'S CUT  *(About)*

Built by **Lohith M** as part of a deep learning learning journey — from raw text all the way to a live deployed web app.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/lohithmothu)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YOUR_USERNAME)

---

<div align="center">

**⭐ If you enjoyed CineScope, give it a star — it's the ultimate 5-star review.**

*Made with 🍿 and Deep Learning*

</div>
