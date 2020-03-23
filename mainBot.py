import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

def yolo():
    im = cv2.imread('room_ser.jpg')
    bbox, label, conf = cv.detect_common_objects(im)

    return label

stopwords = stopwords.words('english')

sentences = [
    'Take a picture',
    'Analyse the picture',
    'How many people are there',
    'Detect faces'
]

def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])

    return text


query = ""

while True:

    print("Type quit to quit!")
    query = input("Query: ")
    sentences.append(query)
    if query == 'quit':
        print()
        break
    
    cleaned = list(map(clean_string, sentences))
    #print(cleaned)

    vectorizer = CountVectorizer().fit_transform(cleaned)
    vectors = vectorizer.toarray()

    csim = cosine_similarity(vectors)
    length = len(csim)

    accuracy = 0.0
    index = -1
    for i in range(length-1):
        temp_accuracy = csim[i][length-1]*100
        temp_accuracy = round(temp_accuracy,1)
        if temp_accuracy > accuracy:
            accuracy = temp_accuracy
            index = i
    
    #print(accuracy,'% , index: ',index)
    if index==1:
        labels = yolo()
        
        print(labels)


    print()
#    print(csim)

