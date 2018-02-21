import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = []
datas.append(pd.read_csv('/home/alex/work/wf/nielsen/UC2/ES_BEER/model/filterTest/stp_filter_size_1_2/stats/details_filter_UC2_ES_BEER.csv', index_col=False))
datas.append(pd.read_csv('/home/alex/work/wf/nielsen/UC2/BE_BEER/model/filterTest/stp/stats/details_filter_UC2_BE_BEER.csv', index_col=False))
datas.append(pd.read_csv('/home/alex/work/wf/nielsen/UC2/DE_AIR_CARE/model/filterTest/stp/stats/details_filter_UC2_DE_AIR_CARE.csv', index_col=False))

fig = plt.figure()
# _1 = fig.add_subplot(2, 2, 1)
# fig.add_subplot(2, 2, 2)
# fig.add_subplot(2, 2, 3)
i = 1
for data in datas:
    data = data[['result', 'InFilterResult']]
    data.InFilterResult = data.InFilterResult.round(1)
    TPcount = pd.DataFrame(data[data.result == 'TP'].groupby(['InFilterResult'])['result'].count())
    FPcount = pd.DataFrame(data[data.result == 'FP'].groupby(['InFilterResult'])['result'].count())
    result = TPcount.join(FPcount, how='outer', lsuffix='TP', rsuffix='FP').fillna(value=0)
    # print result

    ind = result.index  # the x locations for the groups
    width = 0.35        # the width of the bars
    TPs = result.resultTP
    FPs = result.resultFP

    # fig, ax = plt.subplots()
    ax = fig.add_subplot(2, 2, i)
    i = i + 1


    rects1 = ax.bar(ind, TPs, width, color='g')
    rects2 = ax.bar(ind + width, FPs, width, color='r')
    # rects1 = plt.bar(ind, TPs, width, color='g', bottom=FPs)
    # rects2 = plt.bar(ind, FPs, width, color='r')

    # add some text for labels, title and axes ticks
    ax.set_title('Counts of TPs and FPs')
    ax.set_xlabel('Percentage of combinations what are in a filter')
    ax.set_ylabel('Counts')
    # ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels(result.index)

    ax.legend((rects1[0], rects2[0]), ('TPs', 'FPs'))
plt.show()
