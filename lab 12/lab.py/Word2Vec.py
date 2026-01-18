# =========================================================
# Word2Vec on Game of Thrones (VS Code ready)
# =========================================================

import os
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Plotly in browser
pio.renderers.default = "browser"

# -------------------------
# Path to your data folder
# -------------------------
DATA_PATH = r"C:\Users\zumer\OneDrive\Desktop\lab working\lab 12\data"

# -------------------------
# Load all .txt files and tokenize sentences
# -------------------------
story = []

if os.path.exists(DATA_PATH):
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_PATH, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    corpus = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="cp1252") as f:
                    corpus = f.read()

            # Simple split by periods for sentences
            for sent in corpus.strip().split('.'):
                tokens = simple_preprocess(sent)
                if tokens:
                    story.append(tokens)

# If no files are found, use a small sample
if len(story) == 0:
    print("No .txt files found in data folder. Using sample text.")
    sample_text = """
    Daenerys went to Dragonstone. Jon Snow arrived at Winterfell.
    Arya Stark trained with the Faceless Men. Tyrion advised Daenerys.
    Cersei plotted in King's Landing. Bran saw visions of the past.
    """
    for sent in sample_text.strip().split('.'):
        tokens = simple_preprocess(sent)
        if tokens:
            story.append(tokens)

print(f"Number of sentences loaded: {len(story)}")
print("First 2 tokenized sentences:", story[:2])

# -------------------------
# Train Word2Vec Model
# -------------------------
model = Word2Vec(
    vector_size=100,  # word vector dimensions
    window=10,        # context window size
    min_count=1,      # include all words
    sg=0              # 0 = CBOW, 1 = Skip-Gram
)

model.build_vocab(story)
model.train(story, total_examples=model.corpus_count, epochs=model.epochs)

# -------------------------
# Word Embedding Exploration
# -------------------------
def safe_most_similar(word, topn=5):
    if word in model.wv:
        return model.wv.most_similar(word, topn=topn)
    else:
        return f"'{word}' not in vocabulary"

print("Most similar to 'daenerys':", safe_most_similar('daenerys'))

# Odd-one-out
try:
    print("Odd-one-out test:", model.wv.doesnt_match(['jon','arya','cersei','tyrion']))
except ValueError as e:
    print("Odd-one-out error:", e)

# Similarity check
if 'arya' in model.wv and 'sansa' in model.wv:
    print("arya vs sansa similarity:", model.wv.similarity('arya','sansa'))
else:
    print("arya or sansa not in vocabulary")

# -------------------------
# PCA for 3D visualization
# -------------------------
vectors = model.wv.vectors  # get vectors directly
words = model.wv.index_to_key

pca = PCA(n_components=3)
X = pca.fit_transform(vectors)

# Select first 50 words to plot
df = pd.DataFrame(X[:50], columns=['x','y','z'])
df['label'] = words[:50]

fig = px.scatter_3d(df, x='x', y='y', z='z', text='label', color='label')
fig.update_traces(marker=dict(size=5))
fig.show()
