import requests
from bs4 import BeautifulSoup
import re


def req(url):
    r = requests.get(url)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return r.text


def tag_attributes(html):
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.a
    # attributes of tag
    print(tag.name)
    print(tag.string)
    print(tag.attrs)

    # extract value
    print(tag.attrs['class'])


def travel(html):
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.li
    # downward
    print('Tag children:\n',tag.contents)  # .contents[0]
    for child in tag.children:
        print(child)
    # upward
    print('Tag parent:\n', tag.parent)
    # tag.parents
    # parallel
    print('Tag next sibling:\n', tag.next_sibling.next_sibling.prettify())
    print('Tag next sibling:\n', tag.previous_sibling)
    # .next_siblings, .previous_siblings


def extract_info(html):
    soup = BeautifulSoup(html, "html.parser")
    print('Find tag a\n')
    for link in soup.find_all('a'):
        print(link)
    print('Find tag p with attribute:\n')
    for link in soup.find_all('a', 'reference'):
        print(link)
    print('Find tag by id')
    for link in soup.find_all(id="back-top"):
        print(link)
    # .find(), .find_parents, find_parent, ...


# url = "http://www.zuihaodaxue.cn/zuihaodaxuepaiming2016.html"
# r = req(url)

path = 'scikit-learn.html'
html = open(path, 'r', encoding= 'utf-8')
r = html.read()

extract_info(r)


