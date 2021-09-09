import os
path = 'C:/OpenCV/x64/vc16/lib'
dir = os.listdir(path)
fopen = open('C:/OpenCV/x64/vc16/lib/filename.txt','w')
rule = '.lib'
for d in dir :
    if d.endswith(rule):
        string = d + '\n'
        fopen.write(string)
fopen.close()