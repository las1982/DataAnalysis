import sys, os
from bs4 import BeautifulSoup
import fnmatch
from collections import defaultdict

if len(sys.argv) < 2:
    print('Run using command "python ./countGoldTags.py <path/to/dir/with/data>"')
    sys.exit(1)
inpath = sys.argv[1]
flist = fnmatch.filter(os.listdir(inpath), '*.html')
globalTagList = defaultdict(float)
multipleTagSet = set()
for i in range(len(flist)):
    if i % 50 == 0:
        print("Processing " + flist[i])
    with open(inpath + "/" + flist[i], 'r') as fin:
        localTagList = defaultdict(int)
        tagOrders = defaultdict(set)
        html = BeautifulSoup(fin, 'lxml')
        for tag in html.findAll():
            cls = tag.get('class')
            if cls and cls[0] == 'extraction-tag':
                localTagList[tag.name] += 1
                tagOrders[tag.name].add(tag.get('tagorder'))
        for tagName in localTagList:
            globalTagList[tagName] += 1
            if len(tagOrders[tagName]) > 1:
                multipleTagSet.add(tagName)

print("||TAG||\% docs||multiple||")
for w in sorted(globalTagList, key=globalTagList.get, reverse=True):
    print('|%s|%f|%r|' % (w, globalTagList[w] / len(flist), w in multipleTagSet))

print('Tags with multiple values per document:')
print(multipleTagSet)

# import operator
# for k,v in sorted(globalTagList.iteritems(), key=operator.itemgetter(1), reverse=True):
#  print '%s: %f' % (k, v/len(flist))
