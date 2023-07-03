import os

path='/home/tangjl/OWOD/datasets/VOC2007/'
files = open(path+'ImageSets/Main/t2_ft.txt')
line=files.readline().replace('\n','')
while line:
  if not os.path.exists(path+'JPEGImages/'+line+'.jpg'):
    print(line)
  line=files.readline().replace('\n','')
print('done')
files.close()
