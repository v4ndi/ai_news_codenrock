import re
import nltk
import string
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('russian'))

def preprocessing_text(text: str) -> str:
    if type(text) != str:
        return ''
    # text to lower case
    text = text.lower()
    # Remocal english words
    text = output_string = re.sub(r'[a-zA-Z]', '', text)
    # remove punctuation
    text = text.replace('\n', ' ').replace('—', '').replace('«', '')
    PUNCT_TO_REMOVE = string.punctuation
    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    # Removal of stopwords and lemmatize
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
    #Removal of Emojis
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)

    return text

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    answer = pd.DataFrame()
    for category in set(data['category']):

        vectorizer = TfidfVectorizer()
        data_category = data.loc[data['category'] == category]

        X = vectorizer.fit_transform(data_category['text'])
        similarity_matrix = cosine_similarity(X)
        threshold = 0.95

        duplicates = set()
        n = X.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] > threshold:
                    duplicates.add(j)

        for i in range(data_category.shape[0]):
            if i not in duplicates:
                answer = pd.concat([answer, data_category.iloc[[i]]])
    
    return answer

def id_to_label(id: int) -> str:
    dict_id_label = {0: 'Общее',
        1: 'Технологии',
        2: 'бизнес и стартап',
        3: 'блоги',
        4: 'видео и фильмы',
        5: 'дизайн',
        6: 'еда и кулинария',
        7: 'здоровье и медицина',
        8: 'игры',
        9: 'искусство',
        10: 'крипта',
        11: 'маркетинг',
        12: 'мода и красота',
        13: 'музыка',
        14: 'новости и сми',
        15: 'образование',
        16: 'политика',
        17: 'право',
        18: 'психология',
        19: 'путеш',
        20: 'развлечения',
        21: 'рукоделие',
        22: 'софт и приложения',
        23: 'спорт',
        24: 'финансы',
        25: 'фото',
        26: 'цитаты',
        27: 'шоу бизнес',
        28: 'экономика'}

    return dict_id_label[id]

