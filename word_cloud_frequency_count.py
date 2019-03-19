    
import pandas as pd
from nltk import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud

chunk_size=1000
chunks=[]

for chunk in pd.read_csv('control_set.csv', chunksize=chunk_size,lineterminator=' '):    
    chunks.append(chunk)
    


df_control=pd.concat(chunks,axis=0)

del chunk
del chunks



chunks=[]

for chunk in pd.read_csv('en_7.csv', chunksize=chunk_size,lineterminator=' '):    
    chunks.append(chunk)
    


df_purged=pd.concat(chunks,axis=0)

del chunk
del chunks

from nltk.corpus import stopwords
#from string import punctuation

import warnings
warnings.filterwarnings("ignore")

from nltk.tokenize import TweetTokenizer
import re
import string

punctuation = list(string.punctuation)
stop_words = stopwords.words('english') + punctuation + ['rt', 'via',"i'm","don't"]


tokenizer = TweetTokenizer()

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'

    
def tweet_tokenizer(verbatim):
    try:
        stripped = re.sub(combined_pat, '', verbatim)
        stripped = re.sub(www_pat, '', stripped)
        lower_case = stripped.lower()
        letters_only = re.sub("[^a-zA-Z]", " ", lower_case)
    
        all_tokens = tokenizer.tokenize(letters_only)
        
        # this line filters out all tokens that are entirely non-alphabetic characters
        filtered_tokens = [t for t in all_tokens if t.islower()]
        # filter out all tokens that are <2 chars
        filtered_tokens = [x for x in filtered_tokens if len(x)>1]
        
        filtered_tokens = [term for term in filtered_tokens if term not in stop_words]
        
        
    except IndexError:
        filtered_tokens = []
    return(filtered_tokens)
    

    
all_text_control=[]
all_str_control=' '
for i,r in df_control.iterrows():
    pro_text=tweet_tokenizer(r['description'])
    pro_text1=" ".join(pro_text)
    all_text_control.append(pro_text1)
    all_str_control=all_str_control+pro_text1+' '
    
    
all_text_purged=[]
all_str_purged=' '
for i,r in df_purged.iterrows():
    pro_text=tweet_tokenizer(r['description'])
    pro_text1=" ".join(pro_text)
    all_text_purged.append(pro_text1)
    all_str_purged=all_str_purged+pro_text1+' '
    
    
FD_purged=FreqDist(all_str_purged.split())

FD_control=FreqDist(all_str_control.split())

control_T1000=FD_control.most_common(1000)
purged_T1000=FD_purged.most_common(1000)
control_T1000_WL=[x[0] for x in control_T1000]
purged_T1000_WL=[x[0] for x in purged_T1000]

purged_T1000_dict=dict(purged_T1000)
control_T1000_dict=dict(control_T1000)

wordcloud_purged = WordCloud(max_font_size=80, max_words=150, background_color="white",\
                      width=800, height=600)\
                      .generate_from_frequencies(purged_T1000_dict)
plt.figure()
plt.imshow(wordcloud_purged, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud_purged.to_file("WC_purged_T1000.png")

wordcloud_control = WordCloud(max_font_size=80, max_words=150, background_color="white",\
                      width=800, height=600)\
                      .generate_from_frequencies(control_T1000_dict)
plt.figure()
plt.imshow(wordcloud_control, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud_control.to_file("WC_control_T1000.png")

total_word_no_control=len(all_str_control)
total_word_no_purged=len(all_str_purged)

purged_T1000_dict.update((x, float(30000000*y/total_word_no_purged)) for x, y in purged_T1000_dict.items())

control_T1000_dict.update((x, float(30000000*y/total_word_no_control)) for x, y in control_T1000_dict.items())