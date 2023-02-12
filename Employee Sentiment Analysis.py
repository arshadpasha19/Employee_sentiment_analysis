#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("glassdoor_reviews.csv")


# In[2]:


data.head()


# In[3]:


df=data[["firm","work_life_balance","culture_values","career_opp","senior_mgmt","overall_rating","headline","pros","cons"]]


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.dropna(subset=["headline","cons"],inplace=True)


# In[8]:


for col in ['work_life_balance', 'culture_values', 'career_opp', 'senior_mgmt']:
    df[col].fillna(df.groupby("firm")[col].transform("mean"), inplace=True)


# In[9]:


df


# In[10]:


df.isnull().sum()


# In[11]:


df["review"]=df["headline"]+df["pros"]+["cons"]
df.head()


# In[12]:


df_grouped = df[["firm","work_life_balance","culture_values","career_opp","senior_mgmt","overall_rating"]].groupby('firm').mean()


# In[13]:


df_grouped.head()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x='work_life_balance', y='overall_rating', data=df_grouped, alpha=0.5)
sns.scatterplot(x='culture_values', y='overall_rating', data=df_grouped, alpha=0.5)
sns.scatterplot(x='career_opp', y='overall_rating', data=df_grouped, alpha=0.5)
sns.scatterplot(x='senior_mgmt', y='overall_rating', data=df_grouped, alpha=0.5)

plt.xlabel('Work-Life Balance / Culture Values / Career Opportunities / Senior Management')
plt.ylabel('Overall Rating')
plt.title('Relationship between Overall Rating and Other Features')

plt.show()


# In[15]:


unique_firm_names = df['firm'].unique().tolist()
unique_firm_names


# In[16]:


companies=['Accenture','Apple','BBC','Deloitte','EY','Facebook','FirstPort','Goldman-Sachs','Google','H-and-M'
        ,'IBM','Indeed','J-P-Morgan','LinkedIn','Mastercard','McDonald-s','Microsoft','Morrisons',
        'Oracle','Oxford-University','Pizza-Hut','SAP','Tate','VMware','Vodafone','Wipro']


# In[17]:


company_data=df[df['firm'].isin(companies)]


# In[18]:


df = company_data.drop(columns=["headline", "pros", "cons"])
df


# In[ ]:





# In[19]:


import nltk
import string
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('stopwords')


# In[20]:


import re
df['review'] = df['review'].apply(lambda x: re.sub(r'[^\w\s]|\d+', '', x))


# In[21]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from nltk.tokenize import word_tokenize


# In[22]:


df['label'] = np.where(df['overall_rating'] > 3, 1, 0)


# In[23]:


def preprocessor(text):
    # Remove special characters, punctuation, and lowercase the text
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)


# In[24]:


df["review_processed"] = df["review"].apply(preprocessor)


# In[25]:


df


# In[26]:


X = df['review_processed']
y = df['label']


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[29]:


vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[30]:


model = LogisticRegression(max_iter=100)
model.fit(X_train_vectorized, y_train)


# In[31]:


y_pred = model.predict(X_test_vectorized)


# In[32]:


from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test, y_pred))


# In[33]:


data_vectorized=vectorizer.fit_transform(X)


# In[34]:


model.fit(data_vectorized, y)


# In[35]:


pred = model.predict(data_vectorized)


# In[36]:


print("Accuracy: ", accuracy_score(y, pred))


# In[37]:


df['sentiment_score'] = pred


# In[38]:


grouped_df = df.groupby('firm')[["work_life_balance","culture_values","career_opp","senior_mgmt",'label','sentiment_score']]\
            .agg({"work_life_balance":"mean","culture_values":"mean","career_opp":"mean","senior_mgmt":"mean",'label': 'mean','sentiment_score': 'mean'}).reset_index()
grouped_df.rename(columns={'label':'actual_score', 'sentiment_score':'predicted_score'}, inplace=True)


# In[39]:


grouped_df['sentiment_score'] = grouped_df['actual_score'] + grouped_df['predicted_score']+ grouped_df['work_life_balance']\
                                    + grouped_df['culture_values']+ grouped_df['career_opp']+ grouped_df['senior_mgmt']


# In[40]:


grouped_df


# In[41]:


def compare(firm1,firm2):
    score1=grouped_df[grouped_df['firm'] == firm1]['sentiment_score'].values[0]
    score2=grouped_df[grouped_df['firm'] == firm2]['sentiment_score'].values[0]
    print(firm1,score1,firm2,score2)
    if score1>score2:
        print(firm1)
    elif score2>score1:
        print(firm2)
    else:
        print("equal score")
    company_names = [firm1, firm2]
    scores = [score1, score2]

    # Create the bar chart
    plt.bar(company_names, scores)

    # Add labels and title
    plt.xlabel('Company Name')
    plt.ylabel('Score')
    plt.title('Firm Comparision')

    # Show the plot
    plt.show()


# In[42]:


compare("Google","Microsoft")


# In[ ]:





# In[ ]:




