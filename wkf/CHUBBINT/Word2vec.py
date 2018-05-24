import gensim
from gensim.models import word2vec
import re
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('/home/alex/work/projects/DataAnalysis/wkf/CHUBBINT/text8')
sentences = []
with open("/home/alex/work/wf/CHUBBINT/detailed_part_of_body/data/train_to_ngram_sentences.txt") as file:
    for line in file:
        sentences.append(re.split("[\s]+", line.strip()))

# model = word2vec.Word2Vec(sentences, size=300, iter=20, min_count=0, sg=1)
# model.wv.save_word2vec_format('text.model.bin', binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format('text.model.bin', binary=True)

# print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
# print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=2))
# print(model.most_similar(['man']))

model1 = gensim.models.KeyedVectors.load_word2vec_format('text.model.bin', binary=True)
# print(model1.most_similar(['girl', 'father'], ['boy'], topn=3))
vc = model1.wv.get_vector("the_hospital_where")
print(vc, type(vc))

more_examples = ["he is she", "big bigger bad", "going went being"]

for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    # print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))


