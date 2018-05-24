import functools
import sys

items = [1, 2, 3, 4, 5]
squared = []
for x in items:
    squared.append(x**2)
print(squared)

def sqr(x):
    return x**2

print([x for x in map(sqr, items)])

sqr2 = lambda x: x**2
print([x for x in map(sqr2, items)])

print([x for x in map((lambda x: x ** 2), items)])


def square(x):
    return x**2


def cube(x):
    return x**3

funcs = [square, cube]
for r in range(5):
    value = [x for x in map(lambda x: x(r), funcs)]
    print(value)
# sys.exit(0)

print([x for x in filter(lambda x: x >= 0, range(-5, 5))])
print([x for x in range(-5, 5) if x >= 0])
print(functools.reduce(lambda x, y: str(x) + '_*_' + str(y), items))
print(functools.reduce(lambda x, y: x + y, items))

print(items[:-1], items[1:], items[2:])
averages = [(x + y + z) / 2.0 for (x, y, z) in zip(items[:-1], items[1:], items[2:])]
print(averages)
print(sum(items))