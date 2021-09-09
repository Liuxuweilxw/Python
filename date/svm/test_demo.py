import os
import re

# path ='D:/dataset'
# dirs = os.listdir(path)
# for dir in dirs:
#     dirs_nexts = os.listdir('D:/dataset/'+dir+'/out')
#     for dirs_next in dirs_nexts:
#         print(dirs_next)
tks = '2020应战陆路输入疫情——云南瑞丽抗疫七日记'
tks = re.sub(r'[^\u4e00-\u9fa5]', '', tks)
print(tks)