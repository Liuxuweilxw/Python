import urllib
import urllib.request
import urllib.response
import re
import threading
import time
import json
import _thread


# 加载糗事百科
class SpiderModel:
    def __init__(self):

        self.enable = False
        self.pages = []

    #将所有的段子打印出来 添加到列表并返回列表
    def Getpageofkind(self):
        myURL = "http://www.chinanews.com/"
        # print(myURL)
        # 仿冒user_agent 应对网页服务器的反爬虫
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.51"
        # headers = {'User-Agent' : user_agent}
        # 向服务器发送请求
        req = urllib.request.Request(myURL)
        req.add_header("User-Agent", user_agent)
        myResponseofkind = urllib.request.urlopen(req)
        myPageofkind = myResponseofkind.read()
        unicodePageofkind = myPageofkind.decode("utf-8")
        # print(unicodePage)
        # 找出所有class="group"的div标签
        myItemsofkind_first = re.findall('<div class="group">(.*?)</div>', unicodePageofkind, re.S)
        # 去除所有的空格
        for i in range(0, len(myItemsofkind_first)):
            myItemsofkind_first[i] = myItemsofkind_first[i].replace(" ", "")
        # 提取链接
        res_url = r'href="(.*?)"'
        myItemsofkind_final = re.findall(res_url, str(myItemsofkind_first))
        # 只保留www开头的链接
        for i in range(0, len(myItemsofkind_final)):
            myItemsofkind_final[i] = myItemsofkind_final[i].replace('//', "")
            myItemsofkind_final[i] = myItemsofkind_final[i].replace('http:', "")
        myItemsofkind_final.remove('www.chinanews.com/')
        myItemsofkind_final.remove('www.chinanews.com/scroll-news/news1.html')
        myItemsofkind_final.remove('www.chinanews.com/common/news-service.shtml')
        myItemsofkind_final.remove('game.chinanews.com/')
        myItemsofkind_final.remove('www.chinanews.com/photo/')
        print(myItemsofkind_final)

        return myItemsofkind_final

    # def Getpage(self, page, URL):
    #     # 指定网页
    #     myURL = URL
    #     # print(myURL)
    #     # 仿冒user_agent 应对网页服务器的反爬虫
    #     user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.51"
    #     # headers = {'User-Agent' : user_agent}
    #     # 向服务器发送请求
    #     req = urllib.request.Request(myURL)
    #     req.add_header("User-Agent", user_agent)
    #     myResponse = urllib.request.urlopen(req)
    #     myPage = myResponse.read()
    #     unicodePage = myPage.decode("utf-8")
    #     # print(unicodePage)
    #     # 找出所有class="content"的div标签
    #     myItems = re.findall('<div class="content">\n<span>\n\n\n(.*?)\n\n</span>\n\n</div>', unicodePage)
    #     for i in range(0, len(myItems)):
    #         myItems[i] = myItems[i].replace("<br/>", "")
    #     # x = re.findall(r"(?<=\\x01)(.+?)(?=\\)", myItems)
    #     # print(x)
    #     # print(myItems)
    #     return myItems

#     def loadPage(self):
#         while self.enable:
#             try:
#                 if len(self.pages) < 2:
#                     if self.page%15 == 0:
#                         time.sleep(5)
#                     myPage = self.Getpage(self.page)
#                     # print(self.page)
#                     self.pages.append(myPage)
#                     self.page += 1
#                 else:
#                     time.sleep(5)
#             except:
#                 print("无法连接")
#
#
#     def showPage(self, nowPage, page):
#         i = 0
#         for i in range(0, len(nowPage)):
#             if i <len(nowPage):
#                 for content in range(0, len(nowPage[i])):
#                     file_out.write(nowPage[i][content])
#                 file_out.write('\n')
#                 print(('第%d页， 第%d个故事:' % (page,i), nowPage[i]))
#                 i += 1
#             else:
#                 break
#
#
    def start(self):
        self.enable = True
        page = self.page
        print("正在加载中，请稍后")
        #新建一个线程在后台加载段子并存储
        _thread.start_new_thread(self.loadPage,())
        while self.enable:
            # print(page)
            # 如果self的pages数组中存有数据将其输出
            if self.pages:
                print(page)
                nowPage = self.pages[0]
                del self.pages[0]
                self.showPage(nowPage, page)
                page += 1
#
#
myModel = SpiderModel()
# # print(myModel.Getpage(1))
file_out = open("urlofkind.text", 'a', encoding='utf-8')
# 获取类型网页url 写入文件中
htmlodkind =myModel.Getpageofkind()
for i in range(0, len(htmlodkind)):
    file_out.write(htmlodkind[i])
    file_out.write('\n')
file_out.close()

# myModel.start()