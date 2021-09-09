import requests
from bs4 import BeautifulSoup
import re
#自己修改url文件路径
urltxt = open('url/au2.txt', 'r', encoding='UTF-8')
urlList=urltxt.readlines()
i=56443
for url in urlList:
    if(url=='\n'):
        continue
    url=url[:-1]
    print(url)
    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.51"}
    req = requests.get(url, headers=headers)
    #print(head)
    req.encoding = 'utf-8'
    bf = BeautifulSoup(req.text, 'html.parser')
    div = bf.find('div', attrs={'class': 'content'})
    if(div==None):
        continue
    h1 = div.find('h1')
    #print(div)

    if(h1==None):
        continue
    if('�' in h1.get_text()):
        req = requests.get(url, headers=headers)
        req.encoding = 'GB2312'
        bf = BeautifulSoup(req.text, 'html.parser')
        #print(bf)
        div = bf.find('div', attrs={'class': 'content'})
        if (div == None):
            continue
        h1 = div.find('h1')
    print(h1.get_text())
    head = re.sub(r'\s+', '', h1.get_text())
    #修改文字保存路径

    path='C:/Users/liuxuwei/OneDrive/ME/python/database/au2/out/'+str(i)+'.txt'
    print(i)
    i+=1
    out = open(path, 'w', encoding='utf-8', errors='ignore')


    out.write('\n'+head+'\n')

    #timediv = div.find('div', attrs={'class': 'left-t'})
    #time = timediv.get_text().replace(" ", "")[0:16]
    #out.write(time)
    p = div.find('div', attrs={'class': 'left_zw'}).find_all('p')
    for ptext in p:
        out.write('\n'+ptext.text);
    out.close()

