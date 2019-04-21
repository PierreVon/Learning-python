import jieba.analyse

str = "通过用户自定义词典来增强歧义纠错能力"

tags = jieba.analyse.extract_tags(str, withWeight=True)
for tag in tags:
    print("tag: %s\t\t weight: %f" % (tag[0], tag[1]))

print('-------------------------------')
tags = jieba.analyse.textrank(str, withWeight=True)
for tag in tags:
    print("tag: %s\t\t weight: %f" % (tag[0], tag[1]))
