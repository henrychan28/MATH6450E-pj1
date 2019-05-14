import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

number_chars = '0123456789'

with open('ap.txt', 'r') as text:
    data = []
    idx = -1
    for line in text:
        if '<' in line or '>' in line:
            continue
        else:
            data += [line]

processed_data = []

print("#########Start Step 1########")
for idx, datum in enumerate(data):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(datum.lower())
    processed_data += [[word for word in words if word not in stopwords.words('english')]]
    if idx%50==0:
        print("{0}/{1}".format(idx, len(data)))

for idx, processed_datum in enumerate(processed_data):
    remove_single_words = [word.lower() for word in processed_datum if len(word) != 1 and word[0] not in number_chars]
    processed_data[idx] = remove_single_words
    if idx%50==0:
        print("{0}/{1}".format(idx, len(processed_data)))

with open('normalized_tokenized_data.pkl', 'wb') as f:
    pickle.dump(processed_data, f)
