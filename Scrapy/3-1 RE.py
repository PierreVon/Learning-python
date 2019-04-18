import re
macth = re.search(r'[1-9]\d{5}', '100081 NIT')
print(macth.group(0))
print(macth.span())
# .match, .findall, .split, .finditer, .sub

template = re.compile(r'PY.*N')
greedy = template.search('PYANFSDNADNEFN')
print(greedy.group(0))

template = re.compile(r'PY.*?N')
non_greedy = template.search('PYANFSDNADNEFN')
print(non_greedy.group(0))