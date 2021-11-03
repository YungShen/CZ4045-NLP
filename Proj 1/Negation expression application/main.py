import pickle
import re
import string
import nltk
#Uncomment if it is your first time using stopwords
#nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords


def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def stem_text(review):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # remove float prices
    review = re.sub(r'\$[0-9.]+', '', review)
    # remove stock market tickers if the businesses is LISTED
    review = re.sub(r'\$\w*', '', review)
    # remove remaining numbers like dates, scores
    review = re.sub(r'[0-9/]*', '', review)
    # remove hyperlinks
    review = re.sub(r'https?:\/\/.*[\r\n]*', '', review)
    # remove ...
    review = re.sub(r'[.]', '', review)
    # remove hashtags
    # only removing the hash # sign from the word
    review = re.sub(r'#', '', review)
    
    # tokenize review
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    review_tokens = tokenizer.tokenize(review)
    #review_tokens =word_tokenize(review)
    review_clean = []
    for word in review_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            review_clean.append(stem_word)
    return review_clean

def naive_bayes_predict(review, logprior, loglikelihood):
    # process the tweet to get a list of words
    word_l = stem_text(review)
    # initialize probability to zero
    p = 0
    # add the logprior
    p += logprior
    for word in word_l:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]
    return p


if __name__ == "__main__":
    logprior=load_obj("review_word_logprior")['logprior']
    loglikelihood=load_obj("review_word_loglikelihood")
    print('------------------------------------------')
    print('Negation Expression Application')
    print('------------------------------------------')
    
    predict=True
    while (predict):
        print('Welcome to test on our application')
        review_input=input('Enter your review:\n')
        p = naive_bayes_predict(review_input, logprior, loglikelihood)
        if p>0:
            print('This is not a negation expression')
        else:
            print('This is a negation expression!!!')

        next_input=input('Do you want to continue (y/n):')
        if next_input=='n' or next_input=='N':
            predict=False
            print('Exiting')
            print('------------------------------------------')