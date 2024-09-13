#1
test_digit = str(input())
a = test_digit[1:4]
b = a[::-1]
c = test_digit[0] + b + test_digit[-1]
print(c)


#2
days = int(input())
full_weeks = days // 7
residue_week = days % 7
if days == 5:
    target_weekends = 2
elif days == 6:
    target_weekends = 1
else:
    target_weekends = full_weeks * 2
print(target_weekends)


#3
a = int(input())
b = int(input())
c = int(input())
if c % a != 0:
    print(False)
elif c > a*b:
    print(False)
else:
    print(True)


#4
data_to_rome = int(input())
ones = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
tens = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
hunds = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
thous = ["", "M", "MM", "MMM", "MMMM"]
t = thous[data_to_rome // 1000]
h = hunds[data_to_rome // 100 % 10]
te = tens[data_to_rome // 10 % 10]
o = ones[data_to_rome % 10]
print(t+h+te+o)


#5
a = input()
try:
    if float(a) > 0:
        print(True)
except Exception:
    print(False)