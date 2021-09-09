import re
import urllib.request
myURL = "http://www.chinanews.com/"
# print(myURL)
# 仿冒user_agent 应对网页服务器的反爬虫
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.51"
# headers = {'User-Agent' : user_agent}
# 向服务器发送请求
req = urllib.request.Request(myURL)
req.add_header("User-Agent", user_agent)
myResponse = urllib.request.urlopen(req)
myPage = myResponse.read()
unicodePage = myPage.decode("utf-8")
# print(unicodePage)
# 找出所有class="group"的div标签
myItems_first = re.findall('<div class="group">(.*?)</div>', unicodePage, re.S)
# 去除所有的空格
for i in range(0, len(myItems_first)):
    myItems_first[i] = myItems_first[i].replace(" ","")
# 提取链接
res_url=r'href="(.*?)"'
myItems_final = re.findall(res_url, str(myItems_first))
# 只保留www开头的链接
for i in range(0, len(myItems_final)):
    myItems_final[i] = myItems_final[i].replace('//',"")
    myItems_final[i] = myItems_final[i].replace('http:', "")
myItems_final.remove('www.chinanews.com/')
myItems_final.remove('www.chinanews.com/scroll-news/news1.html')
myItems_final.remove('www.chinanews.com/common/news-service.shtml')
myItems_final.remove('game.chinanews.com/')
myItems_final.remove('www.chinanews.com/photo/')
# description = re.findall('www..*?.com/(.*?)/.*?', str(myItems_final))
# description[10] = 'www.chinaqw.com'
# description.insert(11, 'gqqj')
print(myItems_final)
# print(description)
print(len(myItems_final))
# print(len(description))