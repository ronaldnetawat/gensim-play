import gensim.downloader

model = gensim.downloader.load("glove-wiki-gigaword-50")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# get vectors for some related words
words = ["king", "queen", "prince", "princess", "man", "woman", "boy", "girl"]
vectors = [model[word] for word in words]

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(result[:, 0], result[:, 1], c='blue')

for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    
plt.title("Word Embeddings Visualization")
plt.show()