import argparse
I=str
r=None
p=Exception
c=print
A=int
P=argparse.ArgumentParser
import hashlib
S=hashlib.sha256
import librosa
d=librosa.feature
x=librosa.load
import numpy as np
n=np.array
import pandas as pd
g=pd.DataFrame
Y=pd.read_csv
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,StandardScaler
v=['one','two','three','four','five','six','seven','eight','nine']
a="f55ad55fd955e4e760211d4344737f6de1b87722012ec4bea6559fccc418ff04"
W=128
F=44
N=1
def i(M):
 K=Y(M)
 E=[]
 for b,row in K.iterrows():
  M=I(row["file"])
  z=row["digit"]
  e=h(M)
  if e is not r:
   E.append([e,z])
 f=g(E,columns=['feature','class_label'])
 X=f.feature.tolist()
 y=n(f.class_label.tolist())
 le=LabelEncoder()
 y=to_categorical(le.fit_transform(y))
 return X,y
def h(M):
 try:
  O,m=x(M,res_type='kaiser_fast')
  s=d.mfcc(y=O,sr=m,n_mfcc=W)
  H=[]
  for j in s:
   H.append(StandardScaler().fit_transform(j.reshape(-1,1)).reshape(j.shape))
  H=n(H)
 except p as e:
  c("Error encountered while parsing file: ",M,e)
  return r
 if H.shape==(W,F):
  return H
 else:
  return r
def X(x):
 x=n(x)
 x=x.reshape(*x.shape,N)
 return x
def G():
 C=P(description='Hackathon 2020 challenge.')
 C.add_argument('code',type=A,help='The secret code')
 o=C.parse_args()
 if S(I(o.code).encode()).hexdigest()==a:
  c("You've got the code!")
 else:
  c("This is not the code!")
  return
if __name__=='__main__':
 G()
# Created by pyminifier (https://github.com/liftoff/pyminifier)

