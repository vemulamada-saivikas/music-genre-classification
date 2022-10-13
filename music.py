#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function
import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import pickle
import operator


# In[2]:


app = Flask(__name__)

dataset = []
def loadDataset(filename):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

loadDataset("my.dat")


# In[3]:


def distance(instance1 , instance2 , k ):
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    
    #Method to calculate distance between two instances.
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    print("distane is",distance)
    return distance


# In[4]:


def getNeighbors(trainingSet , instance , k):
    distances =[]
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    print("neighbors is ",neighbors)
    return neighbors   


# In[5]:


def nearestClass(neighbors):
    classVote ={}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response]+=1 
        else:
            classVote[response]=1 
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    print("sorter is ",sorter)
    return sorter[0][0]


# In[6]:


print('Model loaded. Check http://127.0.0.1:5000/')


# In[7]:



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('music.html')


# In[10]:


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        
        # Save the file to ./uploads
        basepath = "C:/Users/varun/OneDrive/Desktop/Music-Genre-Classification/Music-Genre-Classification/Flask/"
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        i=1
        
        results = {1: 'blues', 2: 'classical', 3: 'country', 4: 'disco', 5: 'hiphop', 
                   6: 'jazz', 7: 'metal', 8: 'pop', 9: 'reggae', 10: 'rock'}
    
        (rate,sig)=wav.read(file_path)
        print(rate,sig)
        mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature=(mean_matrix,covariance,0)
        pred=nearestClass(getNeighbors(dataset ,feature , 8))
        
        print("predicted genre = ",pred,"class = ",results[pred])
        return "This song is classified as a "+str(results[pred])


# In[ ]:



if __name__ == '__main__':
    app.run(threaded = False)


# In[ ]:




