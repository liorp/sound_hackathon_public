import argparse
ᇇ=str
ﲿ=None
𗜮=Exception
蓰=print
蘒=int
稑=argparse.ArgumentParser
import hashlib
𐬏=hashlib.sha256
import librosa
𧒒=librosa.feature
ﱯ=librosa.load
import numpy as np
𘖼=np.array
import pandas as pd
ࡐ=pd.DataFrame
ݓ=pd.read_csv
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,StandardScaler
ݔ=['one','two','three','four','five','six','seven','eight','nine']
鲬="f55ad55fd955e4e760211d4344737f6de1b87722012ec4bea6559fccc418ff04"
𞤤=128
ݚ=44
𗶧=1
def 𮨷(𐤧):
 辧=ݓ(𐤧)
 迻=[]
 for ﷅ,row in 辧.iterrows():
  𐤧=ᇇ(row["file"])
  𮊍=row["digit"]
  𤵟=𞣂(𐤧)
  if 𤵟 is not ﲿ:
   迻.append([𤵟,𮊍])
 𞺵=ࡐ(迻,columns=['feature','class_label'])
 𐤋=𞺵.feature.tolist()
 겍=𘖼(𞺵.class_label.tolist())
 ޅ=LabelEncoder()
 겍=to_categorical(ޅ.fit_transform(겍))
 return 𐤋,겍
def 𞣂(𐤧):
 try:
  ﰰ,𞺂=ﱯ(𐤧,res_type='kaiser_fast')
  ﹾ=𧒒.mfcc(y=ﰰ,sr=𞺂,n_mfcc=𞤤)
  ཆ=[]
  for 𧎔 in ﹾ:
   ཆ.append(StandardScaler().fit_transform(𧎔.reshape(-1,1)).reshape(𧎔.shape))
  ཆ=𘖼(ཆ)
 except 𗜮 as e:
  蓰("Error encountered while parsing file: ",𐤧,e)
  return ﲿ
 if ཆ.shape==(𞤤,ݚ):
  return ཆ
 else:
  return ﲿ
def ࢪ(𞠲):
 𞠲=𘖼(𞠲)
 𞠲=𞠲.reshape(*𞠲.shape,𗶧)
 return 𞠲
def ﱱ():
 ﱲ=稑(description='Hackathon 2020 challenge.')
 ﱲ.add_argument('code',type=蘒,help='The secret code')
 𠪉=ﱲ.parse_args()
 if 𐬏(ᇇ(𠪉.code).encode()).hexdigest()==鲬:
  蓰("You've got the code!")
 else:
  蓰("This is not the code!")
  return
if __name__=='__main__':
 ﱱ()
# Created by pyminifier (https://github.com/liftoff/pyminifier)

