from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "This is a sample sentence, showing off the stop words filtration."
stop_words = set(stopwords.words('english'))
print('Origin sentence: ', example_sentence)

word_tokens = word_tokenize(example_sentence)
print('Tokens: ', word_tokens)

filtered_sentence = [w for w in word_tokens if not w in stop_words]
print('Tokens without stop word', filtered_sentence)

print('\n', stop_words)

