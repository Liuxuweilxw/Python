import requests
from bs4 import BeautifulSoup
import re
#自己修改url文件路径
urltxt = open(b'G:\data\\fo1.txt', 'r', encoding='UTF-8')
urlList=urltxt.readlines()
for url in urlList:
    if(url=='\n'):
        continue
    url=url[:-1]
    print(url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.51"}
    req = requests.get(url, headers=headers)
    #print(head)
    req.encoding = 'utf-8'
    bf = BeautifulSoup(req.text, 'html.parser')
    div = bf.find('div', attrs={'class': 'content'})
    if(div==None):
        continue
    h1 = div.find('h1')
    print(h1.get_text())
    head = re.sub(r'\s+', '', h1.get_text())
    #修改文字保存路径
    out = open(b'G:\data\est\out.txt', 'a', encoding='GB2312', errors='ignore')


    out.write('\n'+head+'\n')

    timediv = div.find('div', attrs={'class': 'left-t'})
    time = timediv.get_text().replace(" ", "")[0:16]
    out.write(time)
    p = div.find('div', attrs={'class': 'left_zw'}).find_all('p', text=True)
    for ptext in p:
        out.write('\n'+ptext.text);
    out.close()

