from nltk.corpus import wordnet

syns = wordnet.synsets("program")
# only name
print(syns[0].lemmas()[0].name())
print(syns[0].definition())
print(syns[0].examples())
