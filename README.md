# 🎬 IMDB Sentiment Analysis with Deep Learning

Classify IMDB movie reviews as **positive** or **negative** using a variety of deep learning architectures. This project compares traditional Dense networks, CNNs, LSTMs, Bidirectional LSTMs, and Deep CNNs, all leveraging pre‑trained GloVe word embeddings.

## Models Compared
| Model | Test Accuracy |
|-------|---------------|
| Dense (Flatten) | 74.3% |
| CNN (Conv1D) | 84.6% |
| LSTM | 84.3% |
| BiLSTM (Enhanced) | ~87.5% |
| Deep CNN (Enhanced) | ~86.8% |

The **Bidirectional LSTM** with `maxlen=200`, early stopping, and dropout achieves the best performance.

## Setup & Installation
1.Clone the repository
2.Create a virtual environment and activate it
3.Install dependencies:
> pip install -r requirements.txt

## Live Demo
Run the interactive web app locally:
```bash
streamlit run app/demo.py
