import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def stemmer_function(series, stopword_list=None):
    # Lowercases the words and removes non-alphanumeric characters
    series = series.str.lower().str.replace(r"[^a-z0-9'\s]", '', regex=True)
    
    # Tokenizes the words 
    tokenizer = nltk.tokenize.ToktokTokenizer()
    series = series.apply(lambda x: tokenizer.tokenize(x, return_str=True))
    
    # Initializes the PorterStemmer
    ps = nltk.porter.PorterStemmer()
    
    # Uses a list comprehension to stem the words in the series
    stems = [ps.stem(word) for response in series for word in response.split()]
    
    # If no stopword list is provided, use the default NLTK English stopwords
    if not stopword_list:
        stopword_list = stopwords.words('english')
    
    # Filters out the stopwords from the list of stems
    filtered_listed = [w for w in stems if w not in stopword_list]
    
    return filtered_listed
