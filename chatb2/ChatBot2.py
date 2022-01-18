#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# In[2]:


nltk.download('punkt')


# In[3]:


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


# In[4]:


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# In[5]:


def stem(word):
    return stemmer.stem(word.lower())


# In[6]:


def bagOfWords(tokenizeSentence,allWords):
    tokenizeSentence = [stem(w) for w in tokenizeSentence]
    
    bag = np.zeros(len(allWords), dtype = np.float32)
    for idx,w in enumerate(allWords):
        if w in tokenizeSentence:
            bag[idx] = 1.0

    return bag


# In[7]:


a = "How are you"
print(a)
print(tokenize(a))


# In[8]:


word = ['universe','university','universities']

s = [stem(w) for w in word]
print(s)


# In[9]:


s = ["hello","how","are","you"]
w = ["hello","bye","how","see","are","you","soon"]

bagOfWords(s,w)


# In[ ]:





# In[ ]:





# train

# In[10]:


import json


# In[11]:


with open('intents.json','r') as f:
    intents = json.load(f)
    
#print(intents)


# In[12]:


allWords = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        allWords.extend(w)
        xy.append((w,tag))

#print(allWords)
    


# to remove punctuations

# In[13]:


ignoreWords = ['?','!','.',',']
allWords = [stem(w) for w in allWords if w not in ignoreWords]
#print(allWords)


# In[14]:


allWords = sorted(set(allWords))
tags = sorted(set(tags))


# In[15]:


X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bagOfWords(pattern_sentence , allWords)
    X_train.append(bag) 
    
    label = tags.index(tag)
    y_train.append(label)
    
X_train = np.array(X_train)
y_train = np.array(y_train)


# In[16]:


class ChatDataset(Dataset):
       def __init__(self):
           self.n_samples = len(X_train)
           self.x_data = X_train
           self.y_data = y_train
           
       def __getitem__(self, index):
           return self.x_data[index],self.y_data[index]
   
       def __len__(self):
           return self.n_samples
       


# In[17]:


#batch_size = 8

dataset = ChatDataset()
train_loader = DataLoader(dataset, batch_size=8, shuffle=True,num_workers = 0)


# Model

# In[18]:


class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size,num_classes):
        super(NeuralNet , self).__init__()
        self.l1 = nn.Linear(input_size , hidden_size)
        self.l2 = nn.Linear(hidden_size , hidden_size)
        self.l3 = nn.Linear(hidden_size , num_classes)
        
    def forward(self,x):
        out = self.l1(x)
       # out = self.relu(out)
        out = self.l2(out)
        #out = self.relu(out)
        out = self.l3(out)
        
        return out
    


# In[ ]:





# In[19]:


#hyper -para
num_epochs = 1000

learning_rate = 0.001
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])


# In[20]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[21]:


model = NeuralNet(input_size, hidden_size,output_size)


# In[22]:


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device,torch.int64)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')


#print(f'final loss: {loss.item():.4f}')


# In[23]:


data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": allWords,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')


# In[ ]:





# IBM tone analyser

# In[24]:


apikey = 'RvxuZJvwh3us4LY6UZJ5_CqU85WD9dAyA40tdzVI43sr'
url = 'https://api.au-syd.tone-analyzer.watson.cloud.ibm.com/instances/20f7b420-37cf-4b47-91c4-83b3e963b0e4'

from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator = IAMAuthenticator(apikey)
ta = ToneAnalyzerV3(version ='2017-09-21',authenticator = authenticator)
ta.set_service_url(url)


# In[ ]:





# In[ ]:





# In[ ]:





# Lastfm API

# In[25]:


import requests


# In[26]:


API_KEY = '4d040eea4e6701bee1a44157890162ee'
USER_AGENT = 'Roshi_1'


# In[27]:


def lastfm_get(payload):
    # define headers and URL
    headers = {'user-agent': USER_AGENT}
    url = 'https://ws.audioscrobbler.com/2.0/'

    # Add API key and format to the payload
    payload['api_key'] = API_KEY
    payload['format'] = 'json'
    payload['limit'] = 5

    response = requests.get(url, headers=headers, params=payload)
    return response


# To print json file 
# 

# In[28]:


import json                                       #not req for this project

def jprint(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    #print(text)


# In[29]:


def getSongs(tone):
    
    r = lastfm_get({'method': 'tag.getTopTracks','tag':tone})
    
    r0_json = r.json()
    l=[]
    artist1 = r0_json['tracks']['track'][0]['artist']['name']
    for i in range(0,5):
        similar_song = r0_json['tracks']['track'][i]['name']
        
        l.append(similar_song)

    return l,artist1


# In[30]:


l,artist  = getSongs('joy')             #testing
l


# In[31]:


artist                          #testing


# In[32]:


def getSimilarSongs(artist,firstSong):
    
    r = lastfm_get({'method': 'track.getSimilar','artist':artist,'track':firstSong})

    r0_json = r.json()
    l=[]
    for i in range(0,5):
        similar_song = r0_json['similartracks']['track'][i]['name']
        l.append(similar_song)
   
    return l


# In[ ]:





# In[33]:


l = getSimilarSongs('BTS','Butter')           #testing


# In[34]:


l                                           #testing


# In[ ]:





# In[ ]:





# In[43]:


import random 
import torch
import json
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "Bot"
print("Let's chat! (type 'quit' to exit)")
def get_response(msg):
    
    
    # tone analyse
    
    

        #print('songs:',song,'similar:',similarSongs )
    
    sentence = tokenize(msg)
    X = bagOfWords(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
               
    return "I do not understand..."


# In[44]:


def recommend(msg):
    chat = msg
    res = ta.tone(chat).get_result()
    
    if len(res['document_tone']['tones'])!=0:
        tone = res['document_tone']['tones'][0]['tone_name']
        song,artist = getSongs(tone)                    #songs = list of top 5 songs based on tone ,artist=singer of 1st song 
        similarSongs = getSimilarSongs(artist,song[0])   #list of similar songs
        for i in similarSongs:
            song.append(i)
        return song


# In[37]:


recommend('joy')


# In[ ]:


from tkinter import *
#from chat import get_response, bot_name
#7618471758
BG_GRAY = "#CCCC99"
BG_COLOR = "#000000"
TEXT_COLOR = "#EAECEE"
bot_name='bot'
FONT = "Calibri"
FONT_BOLD = "Helvetica 13 bold"

class ChatApplication():
    
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("Chatbot Song Recommender")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=670, height=670, bg=BG_COLOR)
        
        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome to Chatbot Music Recommender System !", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)
        
        # tiny divider
        
        label2 = Label(self.window, bg=BG_GRAY,
                           text="Chat", font=FONT_BOLD, pady=10,relief='solid',borderwidth=4)
        label2.place(relwidth=0.6,relheight=0.07,rely=0.08)

        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=0.6, rely=0.16,relheight=0.01)
        # text widget
       
        self.text_widget = Text(self.window, width=15, height=1, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.650, relwidth=0.6, rely=0.150)
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        
        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.960)
        scrollbar.configure(command=self.text_widget.yview)
        
        label2 = Label(self.window, bg=BG_GRAY,
                           text="Recommendations", font=FONT_BOLD, pady=10,relief='solid',borderwidth=4)
        label2.place(relx=0.6,relwidth=0.4,relheight=0.07,rely=0.08)
    
        self.text_widget2 = Text(self.window, width=15, height=1,bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget2.place(relheight=0.650, relwidth=0.4, relx=0.6, rely=0.150)
        self.text_widget2.configure(cursor="arrow", state=DISABLED)
        
        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.800)
        
        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#000000", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # send button
        send_button = Button(bottom_label, text="Send",fg='#FFFFFF', font=FONT_BOLD, width=20, bg='#996666',
                             command=lambda: self._on_enter_pressed(None),relief='solid',borderwidth=1)
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
     
    def print2(self,similar):
        count=0
        self.text_widget2.configure(state=NORMAL)
        self.text_widget2.delete(1.0,END)
       
        self.text_widget2.insert(END,"\tRecommended\n")
        for i in similar:
            count+=1
            if count==6:
                self.text_widget2.insert(END,"\n\n\n\tSimilar Songs\n")
            if count<=5:
                msg1 = f" {count}:{i} \n"
            else:
                msg1 = f"* {i} \n"
            self.text_widget2.insert(END, msg1)
        self.text_widget2.configure(state=DISABLED)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
        x=recommend(msg)
        if x is not None:
            self.print2(x)
        
        
    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        
        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        
        self.text_widget.see(END)
        #block2
      
             
        
if __name__ == "__main__":
    app = ChatApplication()
    app.run()


# In[ ]:




