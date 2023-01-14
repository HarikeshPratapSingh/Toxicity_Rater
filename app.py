import re
from flask import Flask, request, render_template
import pickle
from bs4 import BeautifulSoup
import lzma
app = Flask(__name__)
with lzma.open('tfidf_comp.xz' , 'rb') as file:
    tfidf_list = pickle.load(file)
with lzma.open('lrs_comp.xz' , 'rb') as file:
    lr_list = pickle.load(file)
with lzma.open('xgbs_comp.xz' , 'rb') as file:
    xgb_list = pickle.load(file)

print(tfidf_list)

def remove_escape(txt): #https://stackoverflow.com/questions/8115261/how-to-remove-all-the-escape-sequences-from-a-list-of-strings
  escapes = ''.join([chr(char) for char in range(0, 32)]+[chr(char) for char in range(127, 161)]) #Escape characters ascii
  translator = str.maketrans('', '', escapes)
  return txt.translate(translator)

def remove_html(txt): #https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
  return BeautifulSoup(txt, "lxml").text

def remove_url(txt): #remove url
  re_url = re.compile(r'https?://\S+|www\.\S+') #Removes urls
  return re_url.sub(r'', txt)


def remove_spl(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', txt)

# https://stackoverflow.com/a/47091490/4084039

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def remove_spaces(txt):
  return re.sub(' +', ' ', txt) #Remove Extra Spaces

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/score',methods=['POST'])
def score():
    text = request.form['u']
    text = remove_url(text)
    text = remove_html(text)
    text = remove_escape(text)
    text = decontracted(text)
    text = remove_spl(text)
    text = text.lower()
    text = remove_spaces(text)
    features1 = tfidf_list[0].transform([text])
    features2 = tfidf_list[1].transform([text])
    features3 = tfidf_list[2].transform([text])
    
    
    prediction = lr_list[0].predict_proba(features1)[:,1] + lr_list[1].predict_proba(features1)[:,1] + lr_list[2].predict_proba(features1)[:,1] + lr_list[3].predict_proba(features1)[:,1] + lr_list[4].predict_proba(features1)[:,1] + lr_list[5].predict_proba(features1)[:,1] +\
                 lr_list[6].predict_proba(features2)[:,1] + lr_list[7].predict_proba(features2)[:,1] + lr_list[8].predict_proba(features2)[:,1] + lr_list[9].predict_proba(features2)[:,1] + lr_list[10].predict_proba(features2)[:,1] + lr_list[11].predict_proba(features2)[:,1] +\
                 lr_list[12].predict_proba(features3)[:,1] + lr_list[13].predict_proba(features3)[:,1] + lr_list[14].predict_proba(features3)[:,1] + lr_list[15].predict_proba(features3)[:,1] + lr_list[16].predict_proba(features3)[:,1] + lr_list[17].predict_proba(features3)[:,1] +\
          1.2 * (xgb_list[0].predict_proba(features1)[:,1] + xgb_list[1].predict_proba(features1)[:,1] + xgb_list[2].predict_proba(features1)[:,1] + lr_list[3].predict_proba(features1)[:,1] + lr_list[4].predict_proba(features1)[:,1] + lr_list[5].predict_proba(features1)[:,1] +\
                 xgb_list[6].predict_proba(features2)[:,1] + xgb_list[7].predict_proba(features2)[:,1] + xgb_list[8].predict_proba(features2)[:,1] + lr_list[9].predict_proba(features2)[:,1] + lr_list[10].predict_proba(features2)[:,1] + lr_list[11].predict_proba(features2)[:,1] +\
                 xgb_list[12].predict_proba(features3)[:,1] + xgb_list[13].predict_proba(features3)[:,1] + xgb_list[14].predict_proba(features3)[:,1] + xgb_list[15].predict_proba(features3)[:,1] + xgb_list[16].predict_proba(features3)[:,1] + xgb_list[17].predict_proba(features3)[:,1])
    output = round(prediction[0]/(18+1.2*18), 2)
    return render_template('index.html', prediction_text='Toxicity Rating {}'.format(output))
    
app.run()