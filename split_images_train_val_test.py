#!/usr/bin/python
import os
import sys
import random
import dircache
import shutil
#getting the parent directory where the different classification image dataset in different folders are present
parent = raw_input ('if ls ../data/ gives you the different categories of image folders then type ../data/ \n')
trpr = float(raw_input ("What is the percentage of file reserved for training in decimal out of 1? \n"))
valpr = float(raw_input ("What is the percentage of file reserved for validation in decimal out of 1? \n"))
tstpr = float(raw_input ("What is the percentage of file reserved for testing in decimal out of 1? \n"))
if (trpr + valpr + tstpr == 1.00):
  #parent = sys.argv[1]
  root, dirs, files = os.walk(parent).next()
  #print root
  #print dirs
  #getting the directory one step up than the parent folder
  pa = root.rsplit('/', 2)
  path = pa[0]
  len_files = []
  
  #getting the number of files present in each folder and saving them
  for i in range (0, len(dirs)):
    path1 = root+dirs[i]
    len_files.append(len(os.listdir(path1)))
  
#getting the number of files for train, val and test
  train = []
  val = []
  test = []
  for i in range (0, len(len_files)):
    t = int(round(len_files[i] * trpr))
    train.append(t)
    v = int(round(len_files[i] * valpr))
    val.append(v)
    test.append(int(len_files[i] - t - v))

#creating the train, val, test
  for i in range (0, len(dirs)):
    cmd_test = "mkdir -p "+path+"/test/"+dirs[i]
    cmd_train = "mkdir -p "+path+"/train/"+dirs[i]
    cmd_val = "mkdir -p "+path+"/val/"+dirs[i]
    if not os.path.exists(cmd_test):
      os.system(cmd_test)
    if not os.path.exists(cmd_train):
      os.system(cmd_train)
    if not os.path.exists(cmd_val):
      os.system(cmd_val)

  for i in range (0, len(dirs)):
    src = root+dirs[i]+"/"
    dest_t = path+"/train/"+dirs[i]+"/"
    dest_v = path+"/val/"+dirs[i]+"/"
    dest_test = path+"/test/"+dirs[i]+"/"

    tr = int(train[i])
    fileName_tr = []
    random.shuffle(dircache.listdir(src))
    fileName_tr = dircache.listdir(src)[:tr]
    res = list(set(dircache.listdir(src))^set(fileName_tr))
    for j in range (0, len(fileName_tr)):
      src_tr = parent+dirs[i]+"/"+fileName_tr[j]
      dst_tr = path+"/train/"+dirs[i]
      cmd = "mv "+src_tr+" "+dst_tr
      os.system(cmd)
  
    va = int(val[i])
    fileName_va = []
    random.shuffle(res)
    fileName_va = res[:va]
    for j in range (0, len(fileName_va)):
      src_va = parent+dirs[i]+"/"+fileName_va[j]
      dst_va = path+"/val/"+dirs[i]
      cmd = "mv "+src_va+" "+dst_va
      os.system(cmd)

    fileName_te = list(set(res)^set(fileName_va))
    for j in range (0, len(fileName_te)):
      src_te = parent+dirs[i]+"/"+fileName_te[j]
      dst_te = path+"/test/"+dirs[i]
      cmd = "mv "+src_te+" "+dst_te
      os.system(cmd)
else:
  print ("Please enter a set of number which will add up to 1.")
