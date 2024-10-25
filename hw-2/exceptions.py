"""
Объявите следующие исключения:
- LowFuelError
- NotEnoughFuel
- CargoOverload
"""

def LowFuelError():
    a = float(input())
    if a < 5:
        raise ValueError('Топлива меньше 5 литров! Заправьте машину!')
    return a
try:
    result = LowFuel()
except ValueError as e:
    print(e)


def NotEnoughFuel():
    a = float(input())
    if a < 1:
        raise ValueError('Топлива меньше 1 литра! Заправьте машину, иначе она никуда не доедет!')
    return a
try:
    result = NotEnoughFuel()
except ValueError as e:
    print(e)

def CargoOverload():
    a = float(input())
    if a > 1000:
        raise ValueError('Слишком большой вес машины с грузом. Необходимо снизить его до 10000 килограмм')
    return a
try:
    result = CargoOverload()
except ValueError as e:
    print(e)


