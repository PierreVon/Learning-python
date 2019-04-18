import requests

url = "http://www.songlyrics.com/taylor-swift/love-story-lyrics/"
r = requests.get(url)
r.raise_for_status()
r.encoding = r.apparent_encoding
print(r.headers)