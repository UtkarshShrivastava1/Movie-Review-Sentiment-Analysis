import pandas as pd
from nltk.corpus import sentiwordnet as swn
import nltk


input_df = pd.read_excel('Input/Review_Input.xlsx')
print('Reading Reviews...')



with open('MasterDictionary/positive_words.txt', 'r') as file:
    positive_words = set(line.strip() for line in file)
with open('MasterDictionary/negative_words.txt', 'r') as file:
    negative_words = set(line.strip() for line in file)

for index, row in input_df.iterrows():
    Review_text = row['Reviews']

    tokens = nltk.word_tokenize(Review_text)

    positive_score = (sum(1 for token in tokens if token in positive_words) / len(tokens)) * 100

    negative_score = (sum(1 for token in tokens if token in negative_words) / len(tokens)) * -100

    intensity_score = 0
    for token in tokens:
        senti_synsets = list(swn.senti_synsets(token))  
        if senti_synsets:
            senti_synset = senti_synsets[0]
            intensity_score += (senti_synset.pos_score() - senti_synset.neg_score())

    intensity_score /= (len(tokens) + 0.000001)

    if intensity_score >= 0.5:
        sentiment_category = 'Very Positive'
    elif intensity_score > 0 and intensity_score < 0.5:
        sentiment_category = 'Positive'
    elif intensity_score == 0:
        sentiment_category = 'Neutral'
    elif intensity_score < 0 and intensity_score > -0.5:
        sentiment_category = 'Negative'
    else:
        sentiment_category = 'Very Negative'
        
    if intensity_score >= 0.75:
        emotion = 'joy'
    elif intensity_score >= 0.25 and intensity_score < 0.75:
        emotion = 'happy'
    elif intensity_score >= -0.25 and intensity_score < 0.25:
        emotion = 'neutral'
    elif intensity_score >= -0.75 and intensity_score < -0.25:
        emotion = 'sad'
    else:
        emotion = 'anger'

    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

    total_words = len(tokens)

    subjectivity_score = (positive_score + abs(negative_score)) / (total_words + 0.000001)

    if polarity_score > 0:
        sentiment = 'positive'
    elif polarity_score < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    
    input_df.at[index, 'SENTIMENT'] = sentiment
    input_df.at[index, 'SENTIMENT CATEGORY'] = sentiment_category
    input_df.at[index, 'EMOTION'] = emotion
input_df.to_excel('Output/Movie_Review_Sentimental_Analysis_Report.xlsx', index=False)
