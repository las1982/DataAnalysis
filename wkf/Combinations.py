import math


def addToOutput(outputArr, ngram):
    outputArr = outputArr
    ngramToAdd = []
    ngramToAdd.extend(ngram)
    outputArr.append(ngramToAdd)
    return outputArr


def position(inpArr, ngram):
    position = len(ngram) - 1
    treshInd = len(inpArr) - 1
    for i in range(len(ngram) - 1, -1, -1):
        el = ngram[i]
        tresh = inpArr[treshInd]
        treshInd = treshInd - 1
        if el + 1 <= tresh:
            position = i
            break
    return position


def increment(ngram, position):
    start = ngram[position]
    for i in range(position, len(ngram), 1):
        start = start + 1
        ngram[i] = start
    return ngram


def num_comb(n, m):
    aa = math.factorial(n)
    bb = math.factorial(n - m)
    cc = math.factorial(m)
    return aa / (bb * cc)


ngramSize = 3
inputsize = 5
inpArr = range(inputsize)
outputArr = []
ngram = range(ngramSize)
# addToOutput(outputArr, ngram)

count = num_comb(inputsize, ngramSize) - 1

while count >= 1:
    pos = position(inpArr, ngram)
    ngram = increment(ngram, pos)
    outputArr.append(ngram)
    # addToOutput(outputArr, ngram)
    count = count - 1

print len(outputArr), outputArr