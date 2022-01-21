'''
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
'''

import streamlit as st

import requests
import json
import re
from bs4 import BeautifulSoup as bs

from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertTokenizer
from textblob import TextBlob
import torch
import textwrap

@st.cache(ttl = 3600)
def load_model( ):
    return DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

@st.cache(ttl = 3600)
def load_tokenizer( ):
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)

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

def scrape_data(productURL):
    # Function to Scrape data 
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
    productPage = requests.get(productURL, headers=headers)
    productSoup = bs(productPage.content,'html.parser')
    
    # Product-Name
    productNames = productSoup.find_all('span', id='productTitle')
    if len(productNames) > 0:
        productNames = productNames[0].get_text().strip()
        #print('.>>> ', productNames)
    
    # Offer-Price
    ids = ['priceblock_dealprice', 'priceblock_ourprice', 'tp_price_block_total_price_ww']
    for ID in ids:
        productDiscountPrice = productSoup.find_all('span', id=ID)
        if len(productDiscountPrice) > 0 :
            productDiscountPrice = productDiscountPrice[0].get_text().strip().split('.')[0]
            productDiscountPrice = productDiscountPrice +'.00'
            break
    #print('.>>> ', productDiscountPrice)
    
    # MRP-Price
    classes = ['priceBlockStrikePriceString', 'a-text-price']
    for CLASS in classes:
        productActualPrice = productSoup.find_all('span', class_=CLASS)
        if len(productActualPrice) > 0 :
            productActualPrice = productActualPrice[0].get_text().strip().split('.')[0]
            productActualPrice = productActualPrice + '.00'
            break
    #print('.>>> ', productActualPrice)
    
    # Product-IMGs
    productImg = productSoup.find_all('img', id="landingImage")
    if len(productImg) > 0:
        productImg = productImg[0]['data-a-dynamic-image']
        productImg = json.loads(productImg)
        #print('.>>> ', productImg)
    
    # Product-Rating
    productRating = productSoup.find_all('span', class_="a-icon-alt")
    if len(productRating) > 0:
        productRating = productRating[0].get_text().strip()
        #print('.>>> ', productRating)

    # Product-Stars
    productStars = productSoup.find_all('table', id="histogramTable")
    if len(productStars) > 0:
        productStars = productStars[0].get_text().replace('\n', '').split('%')
        temp = []
        for i in range(len(productStars)-1):
            temp.append(float(productStars[i][-2:]))
        productStars = temp
        #productStars
    
    # Product-Features
    productFeatures = productSoup.find_all('div', id='feature-bullets')
    if len(productFeatures) > 0:
        productFeatures = productFeatures[0].get_text().strip()
        productFeatures = re.split('\n|  ',productFeatures)
        temp = []
        for i in range(len(productFeatures)):
            if productFeatures[i]!='' and productFeatures[i]!=' ' :
                temp.append( productFeatures[i].strip() )
        productFeatures = temp
        #print('.>>> ', productFeatures)
    
    # Product-Specs
    ids = { 'productDetails_techSpec_section_1' : 'table', 'detailBullets_feature_div' : 'div' }
    for key, value in ids.items():
        productSpecs = productSoup.find_all(value, id=key)
        if len(productSpecs) > 0:
            productSpecs = productSpecs[0].get_text().strip()
            productSpecs = re.split('\n|\u200e|  ',productSpecs) 
            temp = []
            for i in range(len(productSpecs)):
                if productSpecs[i]!='' and productSpecs[i]!=' ' :
                    temp.append( productSpecs[i].strip() )
            productSpecs = temp
            break
    #print('.>>> ', productSpecs)
    
    # Product-Details
    ids = { 'productDetails_db_sections' : 'div' }
    for key, value in ids.items():
        productDetails = productSoup.find_all(value, id=key)
        if len(productDetails) > 0:
            productDetails = productDetails[0].get_text()
            productDetails = re.split('\n|  ',productDetails) 
            temp = []
            for i in range(len(productDetails)):
                if productDetails[i]!='' and productDetails[i]!=' ' :
                    temp.append( productDetails[i].strip() )
            productDetails = temp
            break
    #print('.>>> ', productDetails)
    
    #context1 = productNames + '.\n' + 'M.R.P. : ' + productActualPrice + '. Offer Price : ' + productDiscountPrice # + '. Rating : ' + productRating + '.\n'
    context1 = ''
    for i in range(1, len(productFeatures)-1):
        context1 = context1 + 'Product has ' + productFeatures[i].replace(' | ', ', ') + '. '
    #print('.>>> ', context1)    
    
    context2 = ''
    for i in range(0, len(productSpecs), 2):
        context2 = context2 + productSpecs[i] + ' is ' + productSpecs[i+1] + '. '
    #print('.>>> ', context2)
    
    details = {
        'product_data' : {
            'productNames' : productNames,
            'productDiscountPrice' : productDiscountPrice,
            'productActualPrice' : productActualPrice,
            'productRating' : productRating,
            'productStars' : productStars,
            'productImg' : productImg,
            'productFeatures' : productFeatures,
            'productSpecs' : productSpecs,
            'productDetails' : productDetails,
            'context1' : context1, 
            'context2' : context2 
        }
    }
    return details

def find_answer(answer1, answer2):
    print(answer1, type(answer1))
    answer1 = answer1.split(' ')
    answer2 = answer2.split(' ')
    answer = []

    for word in answer1:
        if not (word in answer):
            answer.append(word)
    answer.append(',')
    for word in answer2:
        if not (word in answer):
            answer.append(word)
    
    answer = " ".join(answer)
    answer = answer.strip().strip(',')
    return answer

def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
    return list

### Code
st.set_page_config(layout='wide')
st.markdown("""
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
""", unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #212121;">
  <a class="navbar-brand" href="https://www.amazon.in/" target="_blank"><b>eSeller</b></a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
</nav>

<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Code
productURL = ''
productName = ''
productID = ''

webPageSideBar = st.sidebar
webPageSideBar.markdown(f'''<h3><b>Ecommerce Websites</b></h3>''', unsafe_allow_html=True)
amazon = webPageSideBar.checkbox('Amazon')
#flipkart = webPageSideBar.checkbox('Flipkart')
productURL_Box = webPageSideBar.empty()

#webPageSideBar.markdown(f'''<h3><b>Code</b></h3>''', unsafe_allow_html=True)
#code = webPageSideBar.checkbox('See Code Here')

if amazon :
    productURL = productURL_Box.text_input('', placeholder="Enter Product Link here")

#if code :
#    st.code('advszvz', language='python')

if(productURL != ''):
    data = scrape_data(productURL)
    col0, col1 = st.columns([30, 70])
    ## Col 0
    with col0:
        imgURL = list(data['product_data']['productImg'].keys())
        st.markdown(f'''<img src={imgURL[0]} alt="product" width="100%" height="100%" style="display: flex; flex-direction:row; justify-content: space-evenly;">
                        <h6>   </h6>
                        <figcaption style="text-align: center;">{data['product_data']['productNames']}</figcaption>''', unsafe_allow_html=True)

    ## Col 1
    with col1:
        product_title = data['product_data']['productNames'].split( '(' )
        MRP = float(re.sub('\D', '', data['product_data']['productActualPrice'][:-2]))
        OfferPrice = float(re.sub('\D', '', data['product_data']['productDiscountPrice'][:-2]))
        UP = sum(data['product_data']['productStars'][0:3])
        DOWN = sum(data['product_data']['productStars'][3:])

        rating = float(data['product_data']['productRating'][0:3])
        ratingHelper = None
        if rating>=0.5 and rating<1.5 :    
            ratingHelper = '<span class="fa fa-star checked"></span>'
        elif rating>=1.5 and rating<2.5 :    
            ratingHelper = '<span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span>'
        elif rating>=2.5 and rating<3.5 :    
            ratingHelper = '<span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span>'
        elif rating>=3.5 and rating<4.5 :
            ratingHelper = '<span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span>'
        else:
            ratingHelper = '<span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span> <span class="fa fa-star checked"></span>'

        st.markdown( f'<h3 style="color:#F7CA00;"> <b>{product_title[0]}</b> </h3>', unsafe_allow_html=True)
        if len(product_title) > 1:
            st.markdown( f'<h5 style="color:#ffffff;"> {"("+product_title[1]} </h5>', unsafe_allow_html=True)
            
        st.markdown(f'''
        <div style="display: flex; flex-direction:row; justify-content: space-evenly;">
            <div class="card bg-dark text-white" style="width: 18rem;">
                <div class="card-body">
                    <h4 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#00C896;"><b>{data['product_data']['productDiscountPrice']}</b></h4>
                    <h4 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#B33F40;"><del>{data['product_data']['productActualPrice']}</del> <b style="color:#00C896;">🡇 {round((MRP-OfferPrice)*100/MRP, 1)}%</b> </h4>
                    <h5 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#F7CA00"><b>Price</b></h5>
                </div>
            </div>
            <div class="card bg-dark text-white" style="width: 18rem;">
                <div class="card-body">
                    <h4 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#B33F40;"> <b style="color:#00C896;">🡅 {UP}%</b>  <b style="color:#B33F40;">🡇 {100-UP}%</b> </h4>
                    <h4 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#FFA41C;"> {ratingHelper} </h4>
                    <h5 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#F7CA00"><b>Rating</b></h5>
                </div>
            </div>
        </div>''', unsafe_allow_html=True)
            
        question = st.text_input('', placeholder="Ask Anything ")
        if question != '':   
            answer1 = qna_bert(data['product_data']['context1'], question)
            answer2 = qna_bert(data['product_data']['context2'], question)
            answer = find_answer(answer1['answer'], answer2['answer'])
            st.success(answer)
else:
    st.markdown(f'''
        <div style="display: flex; flex-direction:row; justify-content: space-evenly;">
            <div class="card bg-dark text-white" style="width: 75rem;">
                <div class="card-body">
                    <h4 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#FFFFFF;"><b>Welcome to <span style="color:#F7CA00;">eSeller</span></b></h4>
                    <h5 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#FFFFFF;"><b>This <span style="color:#F7CA00;">short tutorial</span> will walk you through all of the features of this application.</b></h5>                    
                    <div style="padding:15px;">
                        <img src="https://github.com/TryKode/AQnASystem/blob/main/eSeller.png?raw=true" alt="eSeller" width="100%" height="100%"></img>
                        <h5 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#FFFFFF; position:absolute; top:33%; left:35%; width:325px;"><b>Select a <span style="color:#F7CA00;">Ecommerce Website</span> and put the <span style="color:#F7CA00;">Product Link</span> here.</b></h5>
                    </div>
                    <div style="padding:15px;">
                        <img src="https://github.com/TryKode/AQnASystem/blob/main/eSeller1.png?raw=true" alt="eSeller" width="100%" height="100%"></img>
                        <h5 class="card-subtitle" style="display: flex; flex-direction:row; justify-content: space-evenly; color:#FFFFFF; position:absolute; top:89.5%; left:70%; width:300px;"><b>Verify the Product and <span style="color:#F7CA00;">Ask Any Product related question here</span>.</b></h5>
                    </div>
                </div>
            </div>
        </div>''', unsafe_allow_html=True)
