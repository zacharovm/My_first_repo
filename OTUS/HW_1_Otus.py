test_digit = str(input())
a = test_digit[1:4]
b = a[::-1]
c = test_digit[0] + b + test_digit[-1]
print(c)