#1
def camel_snake_string(target_string: str):
    camelcase_string = ''
    snakecase_string = ''
    for i in target_string:
        if i == '_':
            parts = target_string.split('_')
            for i in parts:
                camelcase_string = camelcase_string + i.capitalize()
            break
        else:
            import re
            t = re.sub(r'([A-Z])', r' \1', target_string).split()
    tt = t[1:]
    for n in range(len(tt)):
        snakecase_string = snakecase_string + '_' + tt[n].lower()
    if camelcase_string != '':
        return camelcase_string
    else:
        return t[0].lower() + snakecase_string

print('Введите строку в формате снейк_кейс или КэмелКейс: ')
a = camel_snake_string(str(input()))
print(a)


#2
def is_leap(year: int):
    return bool(not year % 4 and year % 100 or not year % 400)

def check_date(data: str):
    day, month, year = list(map(int, data.split('.')))
    months = {
        1: 31, 2: 29 if is_leap(year) else 28, 3: 31,
        4: 30, 5: 31, 6: 30,
        7: 31, 8: 30, 9: 30,
        10: 31, 11: 30, 12: 31
    }
    if 0 < year < 10000 and month in months and 0 < day <= months[month]:
        return True
    return False

print('Введите дату: ')
print(check_date(input()))


#3
def plain_digit(a):
    for i in range(2,a):
        if a % i == 0:
            n = False
            break
        else:
            n = True
    return n

print('Введите число: ')
if plain_digit(int(input())) == True:
    print('Это простое число')
else:
    print('Не является простым числом')