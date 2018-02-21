import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print s

dates = pd.date_range('20130101', periods=6)
print dates

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print df

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test_set", "train_set", "test_set", "train_set"]),
                    'F': 'foo'})
print df2
# sys.exit(0)
print df2.dtypes
print df.head()
print df.tail(3)

print df.index
print df.columns
print df.describe()
print df.T

print df
df_tmp = df.sort_index(axis=1, ascending=False)
print df_tmp
df_tmp = df.sort_values(by='B')
print df_tmp

columns = ['A', 'C']
print df['A']
print df[columns][0:3]
print df[0:3]

print df
print df.loc[dates[0]]
print df.loc[dates[:2]]

print df
print df.loc[:, ['A', 'B']]
print df.loc['20130102':'20130104', ['A', 'B']]
print df.loc['20130102', ['A', 'B']]
print df.loc[dates[0], 'A']
print df.at[dates[0], 'A']  # faster

print df
print df.iloc[3]
print df.iloc[3:5, 0:2]
print df.iloc[[1, 2, 4], [0, 2]]
print df.iloc[1:3, :]
print df.iloc[:, 1:3]
print df.iloc[1, 1]
print df.iat[1, 1]  # faster

print df
print df[df.A > 0]
print df[df > 0]

df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print df2
print df2[df2['E'].isin(['two', 'four'])]

print
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
print s1
df['F'] = s1
df.at[dates[0], 'A'] = 0
df.iat[0, 1] = 0
df.loc[:, 'D'] = np.array([5] * len(df))
print df

print
df2 = df.copy()
df2[df2 > 0] = -df2
print df2

print
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
print df1
print df1.dropna(how='any')
print df1.fillna(value=5)
print pd.isnull(df1)

print
print df.mean()
print df
print df.mean(1)

print
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print s
print df.sub(s, axis='index')

print
print df.apply(np.cumsum)
print df.apply(lambda x: x.max() - x.min())

print
s = pd.Series(np.random.randint(0, 7, size=10))
print s
print s.value_counts()

print
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print s.str.lower()

print
df = pd.DataFrame(np.random.randn(10, 4))
print df
pieces = [df[:3], df[3:7], df[7:]]
print pieces[0]
print pieces[1]
print pd.concat(pieces)

print
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
print left
print right
print pd.merge(left, right, on='key')

print
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
print left
print right
print pd.merge(left, right, on='key')

print
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
print df
s = df.iloc[3]
print s
print df.append(s, ignore_index=True)
print df.append(s, ignore_index=False)

print
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})
print df
print df.groupby('A').sum()
print df.groupby('A').count()
print df.groupby(['A', 'B']).sum()

print
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))
print tuples
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
print index
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
print df
df2 = df[:4]
print df2
stacked = df2.stack()
print stacked
print stacked.unstack()
print stacked.unstack(1)
print stacked.unstack(0)

print
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B': ['A', 'B', 'C'] * 4,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D': np.random.randn(12),
                   'E': np.random.randn(12)})
print df
print pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])

rng = pd.date_range('1/1/2012', periods=100, freq='S')
print rng
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print ts
print ts.resample('5Min').sum()

print
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
print ts
ts_utc = ts.tz_localize('UTC')
print ts_utc
print ts_utc.tz_convert('US/Eastern')

print
rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print ts
ps = ts.to_period()
print ps
print ps.to_timestamp()

print
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
print ts.head()

print
df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
print df
df["grade"] = df["raw_grade"].astype("category")
print df["grade"]
df["grade"].cat.categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
print df["grade"]

print
print df.sort_values(by="grade")
print df.groupby("grade").size()

print
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
# ts.plot()
# plt.show()

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
# plt.figure()
# df.plot()
# plt.legend(loc='best')
# plt.show()

print
df.to_csv('foo.csv')
print pd.read_csv('foo.csv')

print
df = pd.DataFrame(
    {u'stratifying_var': np.random.uniform(0, 100, 20),
     u'price': np.random.normal(100, 5, 20)})
df[u'quartiles'] = pd.qcut(
    df[u'stratifying_var'],
    4,
    labels=[u'0-25%', u'25-50%', u'50-75%', u'75-100%'])
# df.boxplot(column=u'price', by=u'quartiles')
# plt.show()

# categorical coorelation
df = pd.DataFrame({'col1': np.random.choice(list('abcde'), 100),
                   'col2': np.random.choice(list('xyz'), 100)}, dtype='category')
df1 = pd.DataFrame({'col1': np.random.choice(list('abcde'), 100),
                    'col2': np.random.choice(list('xyz'), 100)}, dtype='category')

print df
print df1
print
dfa = pd.get_dummies(df)
dfb = pd.get_dummies(df1)
print dfa
print dfb
print
print dfa.corrwith(dfb)



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1, ax2, ax3, ax4]
for n in range(0, len(axs)):
    sample_size = 10 ** (n + 1)
    sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
    print sample
    axs[n].hist(sample, bins=100)
    axs[n].set_title('n={}'.format(sample_size))
plt.show()


