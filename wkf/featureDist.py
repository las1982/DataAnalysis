# Anaylze unbalanced feature distribution
import csv
import json
from collections import defaultdict

targetCol = 'GLOBAL COLLECTION CLAIM'
parentDir = '/media/alex/data/alex/work/wkf/nielsen/UC2/L2G_BE_BEER/unbalanced-test_set/'


def loadInputOutputCols(csvpath, country, category):
    input_columns = []
    output_columns = []
    with open(csvpath, 'rU') as fin:
        csvin = csv.reader(fin)
        headers = csvin.next()
        for row in csvin:
            if row[headers.index('country')] == country and row[headers.index('item_category')] == category:
                input_columns_json = json.loads(row[headers.index('input_columns_json')])
                for data in input_columns_json:
                    input_columns.append(data['label'])
                output_columns_json = json.loads(row[headers.index('output_columns_json')])
                for data in output_columns_json:
                    output_columns.append(data['label'])
    toRemove = []
    for col in input_columns:
        if col in output_columns:
            toRemove.append(col)
    for col in toRemove:
        output_columns.remove(col)
        input_columns.remove(col)
    return input_columns, output_columns


# for each set of features
# build a map of the combination to target column
def getFeatureSetDistribution(csvpath, input_columns):
    ddd = defaultdict(set)
    with open(csvpath, 'rU') as fin:
        csvin = csv.reader(fin)
        headers = csvin.next()
        for row in csvin:
            features = set()
            for col in input_columns:
                if row[headers.index(col)]:
                    features.add(row[headers.index(col)])
            ddd[tuple(features)].add(row[headers.index(targetCol)])
    return ddd


def getPerFeatureDistribution(csvpath, input_col):
    dictFeatVal_trgColVals = defaultdict(set)
    with open(csvpath, 'rU') as fin:
        csvin = csv.reader(fin)
        headers = csvin.next()
        for row in csvin:
            if row[headers.index(input_col)]:
                dictFeatVal_trgColVals[row[headers.index(input_col)]].add(row[headers.index(targetCol)])
    return dictFeatVal_trgColVals

if __name__ == '__main__':
    colInfoPath = parentDir + 'NS_HT_CONFIGURATION.csv'
    inpCols, outCols = loadInputOutputCols(colInfoPath, 'BE', 'BEER')
    trainDataPath = parentDir + 'beer.csv'
    for inp_col in inpCols:
        dictTrgtColVal_featVals = defaultdict(set)
        dictFeatVal_trgColVals = getPerFeatureDistribution(trainDataPath, inp_col)
        for featVal, trgtColVals in dictFeatVal_trgColVals.iteritems():  # key is feature value, value is specific target class output name (set of targetCol values)
            if 'NO CLAIM' not in trgtColVals:
                for trgt_col_val in trgtColVals:
                    dictTrgtColVal_featVals[trgt_col_val].add(featVal)
        if len(dictTrgtColVal_featVals) and inp_col == '#BE LOC DESCF: M#DESCRITPTION':
            for featVal, trgtColVals in dictTrgtColVal_featVals.iteritems():
                print '{}\t{}'.format(featVal, trgtColVals)
            break
            # print '{}\t{}\t{}\t{}'.format(col, len(aaa), sum(len(x) for x in aaa.itervalues()), aaa.keys())


















