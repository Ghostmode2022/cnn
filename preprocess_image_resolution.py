#!/usr/bin/python
import os
import sys
import random
import dircache
import shutil
import cv2

#getting the parent directory where the different classification image dataset in different folders are present
parent = raw_input ('if ls ../data/ gives you the different categories of image folders then type ../data/ \n')

#parent = sys.argv[1]
root, dirs, files = os.walk(parent).next()

for i in range (0, len(dirs)):
  
  src = root+dirs[i]+"/"
  pa = root.rsplit('/', 2)
  path = pa[0]
  print path
  
  cmd_test = "mkdir -p "+path+"/processed_data/test/"+dirs[i]
  cmd_train = "mkdir -p "+path+"/processed_data/train/"+dirs[i]
  cmd_val = "mkdir -p "+path+"/processed_data/val/"+dirs[i]
  if not os.path.exists(cmd_test):
    os.system(cmd_test)
  if not os.path.exists(cmd_train):
    os.system(cmd_train)
  if not os.path.exists(cmd_val):
    os.system(cmd_val)


  dest_tr = path+"/processed_data/train/"+dirs[i]+"/"
  dest_va = path+"/processed_data/val/"+dirs[i]+"/"
  dest_te = path+"/processed_data/test/"+dirs[i]+"/"

  src_tr = path+"/train/"+dirs[i]+"/"
  src_va = path+"/val/"+dirs[i]+"/"
  src_te = path+"/test/"+dirs[i]+"/"

  dirlist_tr = os.listdir(src_tr)
  dirlist_va = os.listdir(src_va)
  dirlist_te = os.listdir(src_te)

  for j in range (0, len(dirlist_tr)):
    image = cv2.imread(src_tr + dirlist_tr[j])
    resized_image = cv2.resize(image, (64,64))
    cv2.imwrite(dest_tr+str(j)+'.jpg', resized_image)

  for j in range (0, len(dirlist_va)):
    image = cv2.imread(src_va + dirlist_va[j])
    resized_image = cv2.resize(image, (64,64))
    cv2.imwrite(dest_va+str(j)+'.jpg', resized_image)

  for j in range (0, len(dirlist_te)):
    image = cv2.imread(src_te + dirlist_te[j])
    resized_image = cv2.resize(image, (64,64))
    cv2.imwrite(dest_te+str(j)+'.jpg', resized_image)

shutil.rmtree(path+"/train")
shutil.rmtree(path+"/val")
shutil.rmtree(path+"/test")

#if not os.path.exists(path+"/train"):
#  os.system(path+"/train")
#if not os.path.exists(path+"/val"):
#  os.system(path+"/val")
#if not os.path.exists(path+"/test"):
#  os.system(path+"/test")

#shutil.move(source, destination)
#shutil.move(path+"/processed_data/train/", path+"/train")
#shutil.move(path+"/processed_data/train/", path+"/val")
#shutil.move(path+"/processed_data/train/", path+"/test")

#shutil.rmtree(path+"/processed_data")
