import urllib.request

myURL = "https://www.qiushibaike.com/hot/page/1/"
        #仿冒user_agent 应对网页服务器的反爬虫
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.51"
# headers={"User-Agent": user_agent}
#向服务器发送请求
myRequest = urllib.request.Request(myURL)
myRequest.add_header("User-Agent", user_agent)
myResponse = urllib.request.urlopen(myRequest)
myPage = myResponse.read().decode('utf-8')
print(myPage)