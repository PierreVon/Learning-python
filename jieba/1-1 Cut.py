import jieba

seg_list = jieba.cut('我是pierre，我来自中国')
print(len(list(seg_list)))
print(','.join(seg_list))

seg_list = jieba.cut('这是一个fashion show', cut_all=True)
print(','.join(seg_list))

