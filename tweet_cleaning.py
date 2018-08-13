
# coding: utf-8

# In[4]:


import json
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


tweets_data_path = '/home/manjula/tweets.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet["text"])
    except:
        continue


# In[6]:


print(len(tweets_data))


# In[7]:


tweets = pd.DataFrame()


# In[8]:


tweets_data


# In[7]:


file2write=open("tweetonly.txt",'w')
#file2write.write(str(tweets_data))
#file2write.close()
for i in range(len(tweets_data)):
    file2write.write(str(tweets_data[i]))
file2write.close()


# In[9]:


import preprocessor as p
c=[]
p.set_options(p.OPT.URL, p.OPT.EMOJI,p.OPT.SMILEY,p.OPT.RESERVED,p.OPT.MENTION,p.OPT.NUMBER)
for i in range(1000):
     x=p.clean(tweets_data[i])
     c.append(x)
print(" \n\n".join(c))

#for i in range(len(c)):


# In[10]:



import re
r=[]
for i in range(len(c)):
    v=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",c[i]).split())
    r.append(v)
print("\n\n".join(r))


# In[11]:


file3=open("cleaned_tweets.csv",'w')
for i in range(len(r)):
    file3.write(str(r[i])+"\n")
file3.close()


# In[12]:


df=pd.read_csv("/home/manjula/cleaned_tweets.csv")

data = {'Tweet': r}


# In[19]:


#df.head()
dff=pd.DataFrame(data)
dff


# In[21]:


dff.columns


# In[1]:


import nltk
from nltk.corpus import stopwords
set(stopwords.words(
    'english'))


# In[24]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#print(r[0])


# In[64]:



stop_words = set(stopwords.words('english'))
token=[]
for i in range(len(r)):
   word_tokens = word_tokenize(r[i])
   #filtered_sentence = [w for w in word_tokens if not w in stop_words]
   #print(word_tokens)
   token.append(word_tokens)
   #To tokenize each tweet
#print(token[0])
filtered=[]
#print(len(token))
for i in range(len(token)):
   filtered_sentence = [w for w in token[i] if not w in stop_words]
   filtered.append(filtered_sentence)
print(filtered)


# In[65]:


from gensim.models import Word2Vec


# In[77]:


model = Word2Vec(filtered, min_count=3,size=200,      # Dimensionality of word embeddings
                 workers=2,     # Number of processors (parallelisation)
                 window=5,      # Context window for words during training
                 iter=30)
print(model)


# In[78]:


words = list(model.wv.vocab)
print(words)


# In[79]:


print(model['Hillary'])


# In[80]:


model.save('model.bin')


# In[81]:


new_model = Word2Vec.load('model.bin')
print(new_model)


# In[82]:


X = model[model.wv.vocab]


# In[83]:


X


# In[88]:


model.most_similar('Hillary')


# In[110]:


import numpy as np
vec=[]
for word in words:
    vector = model[word]
    vec.append(vector)
#arr=np.array(vec)
print(vec)


# In[94]:


from sklearn.decomposition import PCA


# In[95]:


pca = PCA(n_components=2)
result = pca.fit_transform(X)


# In[105]:


plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()

#Do change in figure size

