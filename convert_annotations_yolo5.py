#converts VoTT csv export annotations to YOLO format and modifies to YOLOv5 annotation format
#CHANGE THE TRAIN/TEST/VALID SPLIT (default is 75/20/5)
#instructions:
#place the script one level above 'vott-csv-export'
#run it
#it'll generate a new folder with yolo5 annotation format
#it uses cv2 to get img information

import numpy as np
import pandas as pd
import glob
import os
import shutil
import random

# get vott generated csv
csv_path= glob.glob('./vott-csv-export/*.csv')[0]

#check for multiple files (only 1 should exist)
csv_vott_export= glob.glob('./vott-csv-export/*.csv')
assert len(csv_vott_export) == 1, 'Folder should cointain only 1 csv file'

#get dir full path
thisdir=os.getcwd()

#creates pandas df
annot_df= pd.read_csv(os.path.join(thisdir, csv_path))


#Convert coordinates to yolo
#https://github.com/AlexeyAB/darknet/blob/master/scripts/voc_label.py 
def convert(size, box):
    dw = 1./(size[1]) #cv2 img.shape returns h,w,channels 
    dh = 1./(size[0]) #size = cv2.imread img.shape
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
    
# Define your labels

#manual example
#label_dict= {'angular_ls':0, 'bean_rust':0, 'healthy':1} #possible to group similar classes, could be done in the df to avoid changing here

#automatic class
labels, categories= pd.factorize(annot_df['label'].unique())
#creates dict
label_dict= {}
for i, cat in enumerate(categories):
  label_dict[cat]= labels[i]
  
#convert annotations
#need img sizes for proper convertion
import cv2

def get_img_size(img_path): 
  img= cv2.imread(img_path)
  height, width, _ = img.shape 
  # return width, height #X,Y
  return img.shape

total_imgs= annot_df['image'].unique()

for img_name in total_imgs:
  temp_img_df= annot_df[img_name == annot_df['image']].copy()
  img_shape= get_img_size(os.path.join('./vott-csv-export',img_name))
  # print(temp_img_df)
  with open(f"./vott-csv-export/{temp_img_df['image'][temp_img_df.first_valid_index()][:-4]}.txt", "w") as f:
    for i,row in temp_img_df.iterrows():   
      x,y,w,h = convert(img_shape, [row['xmin'], row['xmax'], row['ymin'], row['ymax']] )   
      f.write(f"{label_dict[row['label']]} {x} {y} {w} {h}\n")

images= sorted(glob.glob('./vott-csv-export/*.jpg'))
if len(images) == 0:
  images= sorted(glob.glob('./vott-csv-export/*.png'))
labels= sorted(glob.glob('./vott-csv-export/*.txt'))


#create dirs functions
def mkdir(dirname):
  import os
  dir= os.path.join(os.getcwd(), dirname)
  if not os.path.exists(dir):
      os.mkdir(dir)
      print('created', dir)
  return

def makedatasetdirs(valid= False):
  dirs= ['train', 'test']
  if valid== True:
    dirs.append('valid')
  for dir in dirs:
    mkdir(dir)
    mkdir(os.path.join(dir, 'images'))
    mkdir(os.path.join(dir, 'labels'))
  return

# make dirs with validation set
makedatasetdirs(valid = True)

#DEFINE YOUR DATA SPLIT 
data_split= {'train': 75, 'valid': 20, 'test': 5}

#Sample the sets
random_list= random.sample(range(len(images)), len(images))

sets_list=[]
for index in random_list:
  sets_list.append([images[index],labels[index]])

train_setsize= (int(np.round_(len(sets_list)*data_split['train']/100,0)))
valid_setsize= (int(np.round_(len(sets_list)*data_split['valid']/100,0)))
test_set_size= (int(np.round_(len(sets_list)*data_split['test']/100,0)))

train_set= sets_list[0:train_setsize]
valid_set= sets_list[train_setsize:(train_setsize+valid_setsize)]
test_set= sets_list[(train_setsize+valid_setsize):]
print(len(train_set), len(valid_set), len(test_set), (len(train_set)+ len(valid_set)+ len(test_set)))

#move files to created dirs
selected_set= iter([train_set, valid_set, test_set])

for set_type in data_split:
  imgs_dest= f'./{set_type}/images'
  labels_dest= f'./{set_type}/labels'

  for image_path, label_path in next(selected_set):
    im_dest= os.path.join(imgs_dest,os.path.basename(image_path))
    os.rename(image_path, im_dest)

    lb_dest= os.path.join(labels_dest,os.path.basename(label_path))
    os.rename(label_path, lb_dest)

# make data.yaml
with open(f"data.yaml", "w") as f:
  f.write('train: ../train/images\n')
  f.write('val: ../valid/images\n')
  f.write('test: ../test/images\n')
  f.write('\n')
  f.write(f'nc: {len(label_dict)}\n') 
  f.write(f"names: {list(label_dict)}")
  
# move to folder

dataset_name= 'dataset_annotated'
mkdir(dataset_name)

shutil.move('./train/','./dataset_annotated')
shutil.move('./valid/','./dataset_annotated')
shutil.move('./test/','./dataset_annotated')
shutil.move('./data.yaml','./dataset_annotated')

