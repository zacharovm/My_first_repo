#1
print('Введите целое число:')
a = str(input())
b = 0
for i in a:
    b = b+int(i)
c = 0
for j in str(b):
    c = c + int(j)
d = 0
if c > 9:
    for k in str(c):
        d = d+int(k)
        print(d)
else:
    print(c)


#2
list_row = [[0,1,1,0], [1, 0, 0, 0], [0,1,0,0]]
tickets = 2
repeat_zero = 0
a = []
for i in list_row:
    for j in i:
            if j == 0:
                repeat_zero = repeat_zero + 1
                if repeat_zero > tickets:
                    a.append(list_row.index(i))
                    break
            else:
                repeat_zero = 0
try:
    print(min(a))
except ValueError:
    print(False)


#3
print('Введите строку:')
input_string = str(input())
encoded_string = []
current_char = input_string[0]
count = 1
for char in input_string[1:]:
    if char != current_char:
        encoded_string.append(str(count) + current_char)
        current_char = char
        count = 1
    else:
        count += 1
encoded_string.append(str(count) + current_char)
print(''.join(encoded_string))


#4
