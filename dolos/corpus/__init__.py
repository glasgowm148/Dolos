import asyncio

import csv


match = False


def analyse(url):

    file1 = open(
        '/Users/pseudo/Documents/GitHub/hci-ae/cogs/dolos/data/guard.csv', 'r')

    myfile = csv.reader(file1)
    for row in myfile:
        if '\n'.join(row).lower() == url.split('/')[2] or "www." + '\n'.join(row).lower() == url.split('/')[2]:
            string = "⚠️⚠️ This domain is a **red-rated** site with false claims about the coronavirus \n Please visit https://www.newsguardtech.com/coronavirus-misinformation-tracking-center/ for more information"
            return string
                

    file1 = open(
        '/Users/pseudo/Documents/GitHub/hci-ae/cogs/dolos/data/guard2.csv', 'r')
    #print("test")
    myfile = csv.reader(file1)
    for row in myfile:
        #print(url.split('/')[2])
        #print('\n'.join(row))
        if '\n'.join(row).lower() == url.split('/')[2]:
            string = "⚠️⚠️ This domain is a **red-rated** site"
            return string


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
            #print(row)
            urlClasses[row[0]] = row[1:]
    try:
        classification = ", ".join(urlClasses[baseUrl])
        print(classification)
        print(urlClasses[baseUrl])
        print('\nclassification according to OpenSources.co:', classification)
    except:
        print('\nsite isnt listed in OpenSources.co')


    print("\nused words are most similar to:   (in descending order)")
    # printClosestMatch(articleInfo['words'], articleInfo, bigrams=bigrams)
   