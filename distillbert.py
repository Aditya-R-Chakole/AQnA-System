import streamlit as st
import streamlit.components.v1 as components

from bs4 import BeautifulSoup as bs
import json
import requests
import re

from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertTokenizer
from textblob import TextBlob
import torch
import textwrap

from PIL import Image

@st.cache(ttl = 3600)
def load_model( ):
    return DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

@st.cache(ttl = 3600)
def load_tokenizer( ):
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
    
def scrape_data(product_url):
    # scrape data
    productURL = product_url
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
    productPage = requests.get(productURL, headers=headers)
    productSoup = bs(productPage.content,'html.parser')

    productNames = productSoup.find_all('span', id='productTitle')
    productNames = productNames[0].get_text().strip()
    
    ids = ['priceblock_dealprice', 'priceblock_ourprice', 'tp_price_block_total_price_ww']
    for ID in ids:
        productDiscountPrice = productSoup.find_all('span', id=ID)
        if len(productDiscountPrice) > 0 :
            break
    productDiscountPrice = productDiscountPrice[0].get_text().strip()
    productDiscountPrice = 'Product Price after Discount '+productDiscountPrice

    classes = ['priceBlockStrikePriceString', 'a-text-price']
    for CLASS in classes:
        productActualPrice = productSoup.find_all('span', class_=CLASS)
        if productActualPrice != [] :
            break
    productActualPrice = productActualPrice[0].get_text().strip()
    productActualPrice = 'Product Actual Price '+productActualPrice

    ###
    productImg = productSoup.find_all('img', alt=productNames)
    productImg = productImg[0]['data-a-dynamic-image']
    productImg = json.loads(productImg)

    productRating = productSoup.find_all('span', class_="a-icon-alt")
    productRating = productRating[0].get_text().strip()
    ###

    productFeatures = productSoup.find_all('div', id='feature-bullets')
    productFeatures = productFeatures[0].get_text().strip()
    productFeatures = re.split('\n|  ',productFeatures)
    temp = []
    for i in range(len(productFeatures)):
        if productFeatures[i]!='' and productFeatures[i]!=' ' :
            temp.append( productFeatures[i].strip() )
    productFeatures = temp
    
    productSpecs = productSoup.find_all('table', id='productDetails_techSpec_section_1')
    productSpecs = productSpecs[0].get_text().strip()
    productSpecs = re.split('\n|\u200e|  ',productSpecs) 
    temp = []
    for i in range(len(productSpecs)):
        if productSpecs[i]!='' and productSpecs[i]!=' ' :
            temp.append( productSpecs[i].strip() )
    productSpecs = temp

    productDetails = productSoup.find_all('div', id='productDetails_db_sections')
    productDetails = productDetails[0].get_text()
    productDetails = re.split('\n|  ',productDetails) 
    temp = []
    for i in range(len(productDetails)):
        if productDetails[i]!='' and productDetails[i]!=' ' :
            temp.append( productDetails[i].strip() )
    productDetails = temp
    
    context = productNames + '\n' + productDiscountPrice + '. ' + productActualPrice + '.\n'
    i = 0
    while i<len(productFeatures):
        context = context + productFeatures[i]+', '
        i = i+1

    i = 0
    while i<len(productSpecs):
        context = context + productSpecs[i]+' '+productSpecs[i+1]+', '
        i = i+2
    context = context[:len(context)-2] + '.\n'

    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> | ', productNames, productDiscountPrice, productActualPrice, productFeatures, productSpecs, productDetails, context, sep="_-_-_-_-_")
    details = {
        'product_data' : {
            'productNames' : productNames,
            'productDiscountPrice' : productDiscountPrice,
            'productActualPrice' : productActualPrice,
            'productRating' : productRating,
            'productImg' : productImg,
            'productFeatures' : productFeatures,
            'productSpecs' : productSpecs,
            'productDetails' : productDetails,
            'context' : context
        }
    }

    return details

def qna_bert(context, question):
    model = load_model()
    tokenizer = load_tokenizer()
        
    def check_spelling(question):
        question = re.sub(r'[^\w\s]', '', question)
        question = question.lower()
        question_list = question.split()

        for i in range(len(question_list)):
            question_list[i] = str( TextBlob(question_list[i]).correct() )
        
        question = " ".join(question_list)
        return (question + " ?")

    def answer_question(question, answer_text):
        encoding = tokenizer.encode_plus(question, answer_text)
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        outputs = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
        answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)

        print ("\nQuestion ",question)
        print ("\nAnswer Tokens: ")
        print (answer_tokens)

        answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

        print ("\nAnswer : ",answer_tokens_to_string)
        return answer_tokens_to_string

    context = context
    question = check_spelling(question)
    answer = answer_question(question, context)

    return {'context': context, 'question' : question, 'answer' : answer}

def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
    return list

### Code 
st.set_page_config(layout='wide')
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #212121;">
  <a class="navbar-brand" href="https://www.amazon.in/" target="_blank"> <b>Amazon</b> Product Question-Answering System</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
</nav>
""", unsafe_allow_html=True)

data = None
page_title = st.markdown(f'<h2 style="color:#df3c83; display: inline-block"><b>Amazon  </b></h2> <h4 style="display: inline-block">Product Question-Answering System</h4>', unsafe_allow_html=True)
product_url_textbox = st.empty()
product_url = product_url_textbox.text_input('', placeholder="Enter Product Link here")

if product_url != '' and not data:
    data = scrape_data(product_url)

    product_title = data['product_data']['productNames'].split( '(' )
    name = product_title[0]
    disc = product_title[1]
    page_title.markdown(f'<h2 style="color:#df3c83; display: inline-block"><b>{name}  </b></h2> <h4 style="display: inline-block">{"(" + disc}</h4>', unsafe_allow_html=True)
    product_url_textbox.empty()

    col0, col1=st.columns([30,70])

    ## Col 0
    with col0:
        imgsUrls = getList( data['product_data']['productImg'] )
        st.image(imgsUrls[0], caption=data['product_data']['productNames'])

    ## Col 1
    with col1:
        rating = float(data['product_data']['productRating'][0:3])

        if rating>=0.5 and rating<1.5 :    
            st.write('** Rating - **', rating, '( :star: )')
        elif rating>=1.5 and rating<2.5 :    
            st.write('** Rating - **', rating, '( :star: :star: )')
        elif rating>=2.5 and rating<3.5 :    
            st.write('** Rating - **', rating, '( :star: :star: :star: )')
        elif rating>=3.5 and rating<4.5 :
            st.write('** Rating - **', rating, '( :star: :star: :star: :star: )')
        else:
            st.write('** Rating - **', rating, '( :star: :star: :star: :star: :star: )')
        
        features = data['product_data']['productFeatures'][0]
        st.markdown( f'<h4 style="color:#df3c83;"> <b>{features}</b> </h4>', unsafe_allow_html=True)
        for pos in range(1, len(data['product_data']['productFeatures'])-1):
            feature = data['product_data']['productFeatures'][pos].split(':')
            st.markdown(f'<h5 style="color:#df3c83; display: inline-block"><b>{feature[0]} - </b></h5> <h6 style="display: inline-block">{feature[1]}</h6>', unsafe_allow_html=True)
                
        question = st.text_input('', placeholder='Ask any Question')
        answer = qna_bert(data['product_data']['context'], question)

        if '[CLS]' in answer['answer'] or '[SEP]' in answer['answer'] and question!='' :
            st.warning('Please Try Changing the Keyword !!!')
            st.warning(answer['answer'])
        elif question!='':
            st.success(answer['answer'])
