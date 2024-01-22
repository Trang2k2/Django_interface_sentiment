
import re
import tensorflow as tf
import re
from underthesea import sent_tokenize, word_tokenize, text_normalize
import emoji_vietnamese as ev


def build_dictionary_from_file(file_path):
    abbreviation_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    abbreviation, full_form = map(str.strip, parts)
                    abbreviation_dict[abbreviation] = full_form
    return abbreviation_dict
# Đường dẫn tới thư mục models
import os

script_directory = os.path.dirname(__file__)
models_directory = 'models'
abbr_dict = build_dictionary_from_file(os.path.join(script_directory, models_directory, 'abbreviate.txt'))
def expand_abbr(text, abbr_dict):
    return ' '.join(abbr_dict.get(word, word) for word in text.split())

def remove_non_alphanumeric(string):
    allowed_characters = r'[^\w\sA-Za-zÀÁẮẤẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴa-z0-9.,\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF.]+'
    return re.sub(allowed_characters, '', string)

def preprocess_text(text, abbr_dict):
    text = expand_abbr(text, abbr_dict)
    text = sent_tokenize(text)
    text = ' '.join(text)
    text = word_tokenize(text, format="text")
    text = text_normalize(text)
    text = ev.demojize(text)
    text = re.sub(r'(:\w+:)', r',\1,', text)
    text = remove_non_alphanumeric(text)
    return text


def get_processed_text(text):
    processed_text = preprocess_text(text, abbr_dict)
    return processed_text

def predict_sentiment_svc(model, text):
    processed_text = preprocess_text(text, abbr_dict)
    print("Process Text:", processed_text)
    sentiment_probabilities = model.predict_proba([processed_text])[0]
    predicted_sentiment_label = model.predict([processed_text])[0]
    return predicted_sentiment_label, sentiment_probabilities


