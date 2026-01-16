import torch
from transformers import AutoModel
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# 1. CHANGE: Detect available GPU and use appropriate device
# Use float32 for maximum compatibility on all chipsets
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v4", 
    trust_remote_code=True,
    torch_dtype=torch.float32 
).to(device)

# Get Text & Image Embeddings for Retrieval
# Data setup
texts = [
    "May the Force be with you",
    "Que la Fuerza te acompañe",
    "フォースと共にあらんことを",
    "Que a Força esteja com você",
    "Möge die Macht mit dir sein",
    "دع القوة تكون معك",
]

image_urls = [
    "https://i.ibb.co/bgBNfMgH/starwars-lightsaber.jpg",
    "https://i.ibb.co/B2bNB4Sd/matrix-code.jpg",
    "https://i.ibb.co/hxJLbTNW/bladerunner-city.jpg",
]

# 2. Inference block
with torch.inference_mode(): # Use inference_mode instead of no_grad for better performance
    print("Encoding texts...")
    text_embeddings = model.encode_text(
        texts=texts,
        task="retrieval",
        prompt_name="query",
        return_numpy=True
    )

    print("Encoding images...")
    # 3. FIX: We remain in float32 to avoid the 'autocast::prioritize' error
    image_embeddings = model.encode_image(
        images=image_urls,
        task="retrieval",
        return_numpy=True
    )

# Compute similarity
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1))

similarities = cosine_similarity(text_embeddings, image_embeddings)

# Display results
print("\nSimilarity Scores (Text vs. Images):")
for i, text in enumerate(texts):
    print(f"\nQuote: {text}")
    for j, url in enumerate(image_urls):
        print(f"  Image {j+1}: {similarities[i][j]:.4f}")

# Visualization
top_image_idx = np.argmax(similarities[0])
try:
    response = requests.get(image_urls[top_image_idx], timeout=10)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.title(f"Top Match for: {texts[0]}")
    plt.axis("off")
    plt.show()
except Exception as e:
    print(f"Could not load image for visualization: {e}")

# Get Text Embeddings for Text Matching
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- A Curated Multimodal, Multilingual Dataset ---
data = {
    "Cosmic Exploration": {
        "texts": ["Nebulae and distant galaxies", "L'exploration spatiale", "星际探索", "Viaje a las estrellas"],
        "images": ["https://i.ibb.co/B5ZNstC9/galaxy.jpg", "https://i.ibb.co/WNBkxR2v/astronaut.jpg"]
    },
    "Deep Sea Biology": {
        "texts": ["Bioluminescent creatures of the deep", "La vida en las fosas abisales", "深海生物", "Tiefseeforschung"],
        "images": ["https://i.ibb.co/LzvH2YKg/jellyfish.jpg", "https://i.ibb.co/yFxk6MrN/anglerfish.jpg"]
    },
    "Architectural Marvels": {
        "texts": ["Modernist architecture", "Готическая архитектура", "未来派の建物", "Ancient Roman structures"],
        "images": ["https://i.ibb.co/zTsKJKLH/modern-architecture.jpg", "https://i.ibb.co/Zp95cW2c/gothic-cathedral.jpg"]
    }
}

# --- Prepare for Embedding ---
all_texts = []
all_images = []
labels = []
for concept, content in data.items():
    all_texts.extend(content["texts"])
    all_images.extend(content["images"])
    labels.extend([concept] * (len(content["texts"]) + len(content["images"])))

# --- Get the Embeddings ---
text_embeddings = model.encode_text(texts=all_texts, task="retrieval", return_numpy=True)
image_embeddings = model.encode_image(images=all_images, task="retrieval", return_numpy=True)

# --- Combine and Reduce Dimensionality ---
all_embeddings = np.concatenate([text_embeddings, image_embeddings])
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
reduced_embeddings = tsne.fit_transform(all_embeddings)

# --- The "Geeky" Visualization ---
plt.figure(figsize=(12, 8))
num_texts = len(all_texts)
colors = {'Cosmic Exploration': 'r', 'Deep Sea Biology': 'b', 'Architectural Marvels': 'g'}

# Plot texts as 'x' and images as 'o'
for i, (embedding, label) in enumerate(zip(reduced_embeddings, labels)):
    marker = 'x' if i < num_texts else 'o'
    plt.scatter(embedding[0], embedding[1], c=colors[label], marker=marker, label=label if i == 0 or labels[i-1] != label else "")

# Create a legend
handles = [plt.Line2D([0], [0], marker='s', color='w', label=concept, markerfacecolor=color, markersize=10) for concept, color in colors.items()]
plt.legend(handles=handles, title="Concepts")
plt.title("2D t-SNE projection of Multimodal, Multilingual Embeddings")
plt.show()


# Get Text Embeddings for Codes Retrieval
ode_snippets = [
    """
def calculate_fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
    """,
    """
import numpy as np
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    """,
    """
import json
def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    """
]

# --- Multilingual "Developer" Queries ---
dev_queries = [
    "une fonction pour sauvegarder des données dans un fichier json", # French: a function to save data to a json file
    "計算フィボナッチ数列の関数", # Japanese: function to calculate the Fibonacci sequence
    "функция для вычисления средней абсолютной ошибки", # Russian: function to calculate mean absolute error
    "eine Funktion, die eine Sequenz von Fibonacci-Zahlen erzeugt" # German: a function that generates a sequence of Fibonacci numbers
]


# Here, we treat code as text, but the principle is the same.
code_embeddings = model.encode_text(texts=code_snippets, task="retrieval", return_numpy=True)
query_embeddings = model.encode_text(texts=dev_queries, task="retrieval", prompt_name="query", return_numpy=True)

# Compute cosine similarity between text and image embeddings
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1))

similarities = cosine_similarity(query_embeddings, code_embeddings)

# Display results
print("Similarity Scores (Text vs. Images):")
for i, text in enumerate(dev_queries):
    print(f"\nQuery: {text}")
    for j, cs in enumerate(code_snippets):
        print(f"  Code snippet {j+1} ({cs}): {similarities[i][j]:.4f}")

# Get Multivectors for Text and Images Retrieval
# We're currently blocked from integrating jina-embeddings-v4 into late interaction retrieval frameworks like PyLate due to some pesky dependency conflicts. If this is a feature you're waiting for, fire up a new issue on our [Hugging Face discussions page]](https://huggingface.co/jinaai/jina-embeddings-v4/discussions/new) to let us know!
texts = [
    "May the Force be with you",  # English
    "Que la Fuerza te acompañe",  # Spanish
    "フォースと共にあらんことを",  # Japanese
    "Que a Força esteja com você",  # Portuguese
    "Möge die Macht mit dir sein",  # German
    "دع القوة تكون معك",  # Arabic
]

# Images of sci-fi movie scenes
images = [
    "https://i.ibb.co/bgBNfMgH/starwars-lightsaber.jpg",  # Star Wars lightsaber duel
    "https://i.ibb.co/B2bNB4Sd/matrix-code.jpg",  # Matrix code rain
    "https://i.ibb.co/hxJLbTNW/bladerunner-city.jpg",  # Blade Runner cityscape
]


multivector_text_embeddings = model.encode_text(
    texts=texts,
    task="retrieval",
    return_multivector=True,
)

multivector_image_embeddings = model.encode_image(
    images=images,
    task="retrieval",
    return_multivector=True,
)