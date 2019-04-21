import jieba.posseg as ps

words = ps.cut('工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作')
for word, flag in words:
    print('%s %s' % (word, flag))
