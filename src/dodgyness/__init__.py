import os
import csv
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
from newspaper import Article
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForQuestionAnswering


def getEmotion(site, plotAttitude=False):
    
    # rate how emotional are the sentences
    sia = SentimentIntensityAnalyzer()
    blob = TextBlob(site['body'])

    compounds = []
    polarity=[]
    subj=[]
    # For each sentence in the site body text
    for sentence in site['sentences']:
        compounds.append(sia.polarity_scores(sentence)['compound'])
        tx= TextBlob(sentence)
        polarity.append(tx.sentiment.polarity)
        subj.append(tx.sentiment.subjectivity)
    compounds = np.array(compounds)
    
    site['emotional charge'] = np.mean(np.abs(compounds) ** 4)
    site['attitude'] = np.mean(compounds ** 3)

    blob.sentiment # polarity : negative/positive feedback; Subjectivity:Opinion provided
    

   
        
        
    poltsent= pd.DataFrame({'polarity':polarity, 'subjectivity':subj})
    #print(poltsent)
    poltsent.plot(title='Polarity & Subjectivity')

    #if plotAttitude:
    return site


def returnKeywords(url):
    sentences = []
    words = []

    # Instantiate a new newspaper Article
    article = Article(url)
    
    # Download the contents
    article.download()

    # Parse
    article.parse()

    # nlp
    article.nlp()

    # Create a new dictionary to store the article info
    articleInfo = {}
    articleInfo['name'] = url

    # Tokenise 
    articleInfo['words'] = word_tokenize(article.text)
    articleInfo['sentences'] = sent_tokenize(article.text)
    
    return articleInfo['words']

  
# This method reads the page into an `Article` format
# https://newspaper.readthedocs.io/en/latest/
def pageReader(url):
    sentences = []
    words = []

    # Instantiate a new newspaper Article
    article = Article(url)
    
    # Download the contents
    article.download()

    # Parse
    article.parse()

    # nlp
    article.nlp()

    # Create a new dictionary to store the article info
    articleInfo = {}
    articleInfo['name'] = url

    # Tokenise 
    articleInfo['words'] = word_tokenize(article.text)
    articleInfo['sentences'] = sent_tokenize(article.text)

    

    # getEmotion()
    articleInfo = getEmotion(articleInfo, plotAttitude=True)

    # TextBlob sentiment
    blob = TextBlob(article.text)
    articleInfo['polarity'] = blob.sentiment.polarity
    articleInfo['sentiment'] = blob.sentiment.subjectivity

    

    compareArticleToOtherSites(articleInfo, False)

    return articleInfo 
    
    

# Chart

def makeGraph2(url):
    article = Article(url)
    article.download()

    # Parse
    article.parse()

    # nlp
    article.nlp()
    blob = TextBlob(article.text)
    # Seaborn

    # configure size of heatmap
    #sns.set(rc={'figure.figsize':(35,3)})

    # function to visualize
    def visualise_sentiments(data):
        svm = sns.heatmap(pd.DataFrame(data).set_index("Sentence").T,center=0, annot=True, cmap = "PiYG")
        image_object = BytesIO()
        figure = svm.get_figure()    
        figure.savefig(image_object, format="PNG", facecolor="#36393E")
        
        #sns.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
        
        image_object.seek(0)
        return image_object


    # visualization
    return visualise_sentiments({
        "Sentence":["SENTENCE"] + blob.split(),
        "Sentiment":[blob.sentiment.polarity] + [blob.sentiment.polarity for word in blob.split()],
        "Subjectivity":[blob.sentiment] + [blob.sentiment for word in blob.split()],
    })

def makePlot(D):
    new_dict = {}

    #new_dict['emotional charge'] = D['emotional charge']
    new_dict['polarity'] = D['polarity']
    new_dict['attitude'] = D['attitude']
    new_dict['sentiment'] = D['sentiment']

    lists = sorted(new_dict.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    plt.plot(x, y)
    plt.show()

    title = plt.title("My Analysis", color="white")
    params = {  
                "ytick.color" : "w",
                "xtick.color" : "w",
                "axes.labelcolor" : "w",
                "axes.edgecolor" : "w" 
            }

    plt.rcParams.update(params)
    

    image_object = BytesIO()
    plt.savefig(image_object, format="PNG", facecolor="#36393E")
    plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
    image_object.seek(0)
    return image_object



def makeGraph(D):

   
    
    #plt.figure()
    #print(*zip(new_dict.keys()))
    #plt.bar(*zip(*new_dict.items()))

    # Set the axis labels to white 
    title = plt.title("My Analysis", color="white")
    params = {  
                "ytick.color" : "w",
                "xtick.color" : "w",
                "axes.labelcolor" : "w",
                "axes.edgecolor" : "w" 
            }

    plt.rcParams.update(params)
    

    image_object = BytesIO()
    plt.savefig(image_object, format="PNG", facecolor="#36393E")
    plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
    image_object.seek(0)
    return image_object

# Save the articleInfo info to a pickle
def save(articleInfo):
    with open('articleInfo.pickle', 'wb') as handle:
        pickle.dump(articleInfo, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the articleInfo info from the pickle
def load():
    path = os.path.dirname(__file__)
    path = os.path.join(path, 'articleInfo.pickle')
    with open(path, 'rb') as handle:
        articleInfo = pickle.load(handle)
    return articleInfo


# Compare against sourcesUncut.csv
def compareArticleToOtherSites(articleInfo, bigrams=False):
    #print(articleInfo)
    if articleInfo == 'DEFAULT':
        articleInfo = load()
    #print("test1")
    path = os.path.dirname(__file__)
    path = os.path.join(path, '../data/sourcesUncut.csv')
    urlClasses = {}
    with open(path, 'r') as csvfile:
        #print("test2")
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            print(row)
            urlClasses[row[0]] = row[1:]
    try:
        classification = ", ".join(urlClasses[baseUrl])
        #print(classification)
        #print(urlClasses[baseUrl])
        #print('\nclassification according to OpenSources.co:', classification)
    except:
        print('\nsite isnt listed in OpenSources.co')


    print("\nused words are most similar to:   (in descending order)")
    # printClosestMatch(articleInfo['words'], articleInfo, bigrams=bigrams)


def printClosestMatch(words, articleInfo, bigrams=False):
    wordLengths = [len(siteInfo['words']) for siteInfo in articleInfo]
    minLength = min(wordLengths)
    distances = []
    for siteInfo in articleInfo:
        distances.append([vocabulariesDistance(words,
                                               siteInfo['words'][:minLength],
                                               bigrams=bigrams),
                          siteInfo['name']])
    for pair in sorted(distances):
        print(round(pair[0], 4), "\t", pair[1])

def vaderAnalysis(articleInfo):
    sia = SentimentIntensityAnalyzer()
    corpus = list(articleInfo['sentences'])
    print("corpus")
    print(corpus)
    sentimentscores = []
    for i in corpus:
        score = sia.polarity_scores(i)
        score['title'] = i
        sentimentscores.append(score)
    
    sentimentdf = pd.DataFrame(sentimentscores)
    sentimentdf.drop(columns=['title'], inplace = True)
    print(sentimentdf)

def nlpQuestion(question):
    print("nlptriggered")
    text = """
    Coronaviruses are a large family of viruses that can cause illness in animals or humans. In humans, several coronaviruses are known to cause respiratory infections ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS) and Severe Acute Respiratory Syndrome (SARS). COVID-19 is a virus of the same family with a first recorded outbreak in Wuhan, China, in December 2019. The most common symptoms of COVID-19 are fever, tiredness, and dry cough. Other symptoms include aches and pains, nasal congestion, runny nose, sore throat or diarrhea. These symptoms are usually mild and begin gradually. Some people become infected but don’t develop any symptoms and don't feel unwell. Most people (about 80%) recover from the disease without needing special treatment. Around 1 out of every 6 people who gets COVID-19 becomes seriously ill and develops difficulty breathing. Older people, and those with underlying medical problems like high blood pressure, heart problems or diabetes, are more likely to develop serious illness. People with fever, cough and difficulty breathing should seek medical attention. People can catch COVID-19 from others who have the virus. The disease can spread from person to person through small droplets from the nose or mouth which are spread when a person with COVID-19 coughs or exhales. These droplets land on objects and surfaces around the person. Other people then catch COVID-19 by touching these objects or surfaces, then touching their eyes, nose or mouth. People can also catch COVID-19 if they breathe in droplets from a person with COVID-19 who coughs out or exhales droplets. This is why it is important to stay more than 1 meter (3 feet) away from a person who is sick.
    Studies to date suggest that the virus that causes COVID-19 is mainly transmitted through contact with respiratory droplets rather than through the air. 
    There have been 105000 confirmed cases of coronovirus in the world, with 3100 deaths. There are only 32 confirmed cases in Lebanon. If you are experiencing symptoms, call MOPH on 1214 or 76592699.
    """
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    print("nlptriggered")

    input_ids = tokenizer.encode(question, text)
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    print("nlptriggered")

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]).replace(' ##', '')


    print(answer)
    return answer


'''
def diversity(words):
    # 0..1 the less the better diversity
    vocabulary = Text(words).vocab()
    sum = 0
    for w in vocabulary.items():
        sum += w[1] ** 2
    return sum / len(words) ** 2


def walkingSum(array, bufferLength):
    # makes the graph smoother
    return [sum(array[i:i + bufferLength])
            for i in range(len(array) - bufferLength + 1)]


def vocabulariesDistance(words1, words2, bigrams=False):
    # measures how different are two sets of words
    # if told to use bigrams substitute bigrams for words array
    if bigrams:
        words1 = [words1[i] + ' ' + words1[i + 1]
                  for i in range(len(words1) - 1)]
        words2 = [words2[i] + ' ' + words2[i + 1]
                  for i in range(len(words2) - 1)]

    allDifferentWords = set(words1).union(words2)
    # vocabulary is a dictionary: word -> number of occurences
    vocabulary1 = Text(words1).vocab()
    vocabulary2 = Text(words2).vocab()

    len1 = len(words1)
    len2 = len(words2)
    distance = 0
    for word in allDifferentWords:
        distance += np.abs(vocabulary1[word] / len1 - vocabulary2[word] / len2)
    return distance












siteNames = ['https://apnews.com/article/7d8b0e32efd0480fbd12acf27729f6a5']





def getArticleUrls(site):
    # turned off momoize because
    # for some reason you cannot cache
    fullSiteBuild = newspaper.build(site, memoize_articles=False)
    return fullSiteBuild.articles


def processArticles(articles, requestedNumberOfWords):
    # download articles and extract sentences and words
    sentences = []
    words = []
    counter = 0
    countProcessedArticles = 0
    while len(words) < requestedNumberOfWords:
        counter += 1
        if articles == []:
            print('no more articles left')
            break
        if counter > 150:
            print("too many sites visited")
            break
        try:
            articles[0].download()
            articles[0].parse()
            articles[0].nlp()
            sentences += sent_tokenize(articles[0].text)
            words += word_tokenize(articles[0].text)
            countProcessedArticles += 1
            print('processed article: ' + articles[0].url)
        except:
            print('couldnt process article: ' + articles[0].url)
        articles.remove(articles[0])
    return [words, sentences, countProcessedArticles]


def mainInformationGatherer(siteNames, articleInfo,
                            requestedNumberOfWords=10000):
    # schedule articles download and processing
    for siteName in siteNames:
        foundSite = False
        # check if info about this site already exists
        for siteInfo in articleInfo:
            if siteInfo['name'] == siteName:
                foundSite = True
                siteToEdit = siteInfo
                break
        if not foundSite:
            siteToEdit = {}
            siteToEdit['name'] = siteName
            siteToEdit['articles'] = getArticleUrls(siteName)
            siteToEdit['words'] = []
            siteToEdit['sentences'] = []
            siteToEdit['processed articles'] = 0
            articleInfo.append(siteToEdit)

        newWords, newSentences, newProcessedArticles =\
            processArticles(siteToEdit['articles'], requestedNumberOfWords)

        siteToEdit['words'] += newWords
        siteToEdit['sentences'] += newSentences
        siteToEdit['processed articles'] += newProcessedArticles
    return articleInfo
'''