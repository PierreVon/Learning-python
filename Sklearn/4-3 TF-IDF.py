from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
     "I don't like your little games",
     "Don't like your tilted stage",
     "The role you made me play of the fool",
     "No I don't like you"
]

vectorizer = TfidfVectorizer()

vectorizer.fit(texts)
print(vectorizer.vocabulary_)
print(vectorizer.idf_)

# calculate tf-idf of texts[0]
vector = vectorizer.transform([texts[0]])
print(vector.shape)
print(vector.toarray())