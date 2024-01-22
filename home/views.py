from cgi import print_directory
import json
import os
import tensorflow as tf
from django.shortcuts import render
from .forms import ReviewForm
from .sentiment_predictor import predict_sentiment_svc, get_processed_text
import numpy as np  # Import NumPy

def home(request):
    return render(request, 'index.html')

def index1(request):
    form = ReviewForm()
    svc_result = None
    svc_probabilities = None
    sentiment_label = "Unknown"  # Set a default value
    processed_text = None

    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review_text = form.cleaned_data['review']
            processed_text = get_processed_text(review_text)

            # Load SVC model
            svc_model = load_svc_model()
            svc_result, svc_probabilities = predict_sentiment_svc(svc_model, review_text)

            # Convert probabilities to sentiment labels
            sentiment_labels = ["Negative", "Positive", "Neutral"]

            # Find the index of the maximum probability
            max_prob_index = np.argmax(svc_probabilities)
            sentiment_label = sentiment_labels[max_prob_index]

            # Print sentiment label and probabilities
            print("Sentiment Label:", sentiment_label)
            print("Probabilities:")
            for i, prob in enumerate(svc_probabilities):
                print(f"{sentiment_labels[i]} - Probability: {prob}")
            
            
    return render(request, 'index.html', {
        'form': form,
        'svc_result': svc_result,
        'svc_probabilities': svc_probabilities,
        'sentiment_label': sentiment_label,
        'processed_text': processed_text,
        #'sentiment_probabilities': zip(sentiment_labels, svc_probabilities),  # Combine labels and probabilities for template
        
    })




import pickle
import os

def load_svc_model():
    # Đường dẫn tới thư mục models
    script_directory = os.path.dirname(__file__)

    # Đường dẫn tới file model SVC
    svc_model_path = os.path.join(script_directory, 'models', 'svc_model.pkl')

    try:
        with open(svc_model_path, 'rb') as file:
            svc_model = pickle.load(file)
        return svc_model
    except FileNotFoundError:
        print(f"File not found: {svc_model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

