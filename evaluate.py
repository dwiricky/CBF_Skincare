import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
ingred_data = pd.read_csv('processed_ingred (1).csv')  # Ingredient data
products_data = pd.read_csv('skincare_products.csv')  # Product data

# Step 1: Preprocessing - Clean and combine the ingredients
products_data['ingredients_list'] = products_data['ingredients'].fillna('')

# Step 2: TF-IDF Vectorization of ingredients
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(products_data['ingredients_list'])

# Step 3: Calculate Cosine Similarity between all products
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 4: K-Means Clustering based on ingredient similarity
num_clusters = 5  # Define number of clusters (can be adjusted)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)
products_data['cluster'] = kmeans.labels_

# Step 5: Visualize the clustering
plt.figure(figsize=(10, 8))
plt.scatter(range(len(products_data)), [0]*len(products_data), c=products_data['cluster'], cmap='viridis', s=50)
plt.title("Clustering of Products Based on Ingredient Similarity")
plt.xlabel("Product Index")
plt.yticks([])
plt.colorbar(label="Cluster")
plt.show()

# Step 6: Recommendation based on clustering (example: first product's cluster)
def get_cluster_recommendations(cluster_id, num_recommendations=5):
    """Get top k products recommended for a given cluster."""
    recommended_products = products_data[products_data['cluster'] == cluster_id]
    return recommended_products['product_name'].head(num_recommendations)

# Example: Get recommendations for the first product's cluster
cluster_id = products_data['cluster'].iloc[0]  # Get the cluster of the first product
recommended_products = get_cluster_recommendations(cluster_id)
print("Recommended Products for Cluster", cluster_id)
print(recommended_products)

# Step 7: Example function for content-based filtering recommendations based on cosine similarity
def get_recommendations(product_index, cosine_sim, num_recommendations=5):
    """Get top k products based on cosine similarity."""
    sim_scores = list(enumerate(cosine_sim[product_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]  # Exclude itself
    product_indices = [i[0] for i in sim_scores]
    return products_data['product_name'].iloc[product_indices]

# Example: Get recommendations for the first product using content-based filtering
recommended_products_cbf = get_recommendations(0, cosine_sim)
print("\nContent-Based Filtering Recommendations for Product 0:")
print(recommended_products_cbf)

# Step 8: Precision at k evaluation
def precision_at_k(recommended_products, actual_products, k=5):
    """Calculate Precision at k."""
    relevant_items = [1 if item in actual_products else 0 for item in recommended_products[:k]]
    return sum(relevant_items) / k

# Step 9: Recall at k evaluation
def recall_at_k(recommended_products, actual_products, k=5):
    """Calculate Recall at k."""
    relevant_items = [1 if item in actual_products else 0 for item in recommended_products[:k]]
    return sum(relevant_items) / len(actual_products)

# Step 10: F1-score at k evaluation
def f1_at_k(recommended_products, actual_products, k=5):
    """Calculate F1-score at k."""
    precision = precision_at_k(recommended_products, actual_products, k)
    recall = recall_at_k(recommended_products, actual_products, k)
    if precision + recall == 0:
        return 0  # Avoid division by zero
    return 2 * (precision * recall) / (precision + recall)

# Step 11: Evaluate multiple products (first 10 products in the dataset)
k = 3  # Top-k recommendations
all_precision = []
all_recall = []
all_f1 = []

# Iterate for the first 10 products (as an example)
for idx in range(10):
    recommended_products = get_recommendations(idx, cosine_sim, num_recommendations=5)
    actual_products = ['The Ordinary Natural Moisturizing Factors + HA', 'The Ordinary Amino Acids + B5']  # Example actual products

    precision = precision_at_k(recommended_products, actual_products, k)
    recall = recall_at_k(recommended_products, actual_products, k)
    f1 = f1_at_k(recommended_products, actual_products, k)
    
    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)

# Plotting the evaluation results (Precision, Recall, and F1-score) for the first 10 products
fig, ax = plt.subplots(figsize=(10, 6))

# Create x axis for the products evaluated
x = list(range(1, 11))  # 10 products evaluated

# Plot each metric
ax.plot(x, all_precision, label="Precision", marker='o')
ax.plot(x, all_recall, label="Recall", marker='o')
ax.plot(x, all_f1, label="F1-score", marker='o')

# Adding labels and title
ax.set_xlabel("Product Index")
ax.set_ylabel("Score")
ax.set_title(f"Precision, Recall, and F1-Score at {k} for First 10 Products")
ax.legend()

# Display the plot
plt.show()
