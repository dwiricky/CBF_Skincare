import random
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


app = Flask(__name__)

# Load datasets
ingred_file_path = 'processed_ingred (1).csv'
products_file_path = 'skincare_products.csv'

ingred_data = pd.read_csv(ingred_file_path)
products_data = pd.read_csv(products_file_path)

# --- Content-Based Filtering Function ---
def content_based_recommendation(user_skin_type, products_data, ingred_data):
    # Filter bahan berdasarkan jenis kulit yang sesuai
    matching_ingredients = ingred_data[ingred_data['Skin Type'].str.contains(user_skin_type, case=False, na=False)]
    
    # Ambil semua bahan yang cocok dengan jenis kulit
    matching_ingredients_list = matching_ingredients['Ingredient'].tolist()
    
    # Filter produk yang mengandung bahan yang cocok dengan jenis kulit
    recommended_products = []
    for idx, row in products_data.iterrows():
        ingredients_in_product = row['ingredients'].split(',')
        if any(ingredient.strip() in matching_ingredients_list for ingredient in ingredients_in_product):
            recommended_products.append(row['product_name'])
    
    # Ambil 5 produk acak jika ada produk yang cocok
    if recommended_products:
        return random.sample(recommended_products, min(5, len(recommended_products)))
    
    return "Tidak ada produk yang sesuai dengan jenis kulit ini."

# --- Evaluate Recommendations ---
def evaluate_recommendations(true_relevant, predicted_relevant):
    """
    Evaluate the recommendations using precision, recall, and f1-score
    :param true_relevant: List of true relevant products (the ground truth)
    :param predicted_relevant: List of predicted relevant products (the system recommendations)
    :return: Precision, Recall, F1-Score
    """
    # True Positive: Items that are in both true and predicted relevant
    true_positive = len(set(true_relevant).intersection(predicted_relevant))
    
    # Precision: True Positives / (True Positives + False Positives)
    precision = true_positive / len(predicted_relevant) if len(predicted_relevant) > 0 else 0
    
    # Recall: True Positives / (True Positives + False Negatives)
    recall = true_positive / len(true_relevant) if len(true_relevant) > 0 else 0
    
    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# --- API Endpoints ---

# Endpoint untuk rekomendasi produk berdasarkan jenis kulit
@app.route('/recommendation', methods=['GET'])
def recommendation():
    user_skin_type = request.args.get('skin_type', default='Sensitive skin', type=str)
    recommended_products = content_based_recommendation(user_skin_type, products_data, ingred_data)
    
    # Misalnya, kita punya data relevansi produk (True relevants) dan hasil rekomendasi
    true_relevant_products = ['product_A', 'product_B', 'product_C']  # Contoh produk yang relevan dari dataset sebenarnya
    predicted_relevant_products = recommended_products  # Produk yang direkomendasikan oleh sistem
    
    # Evaluasi dengan metrik precision, recall, f1-score
    precision, recall, f1 = evaluate_recommendations(true_relevant_products, predicted_relevant_products)
    
    evaluation_results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return jsonify({
        'recommended_products': recommended_products,
        'evaluation_results': evaluation_results
    })

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
