class MediaFiles:

    def __init__(self, name, size, creation_date, owner):
        self.name = name
        self.size = size
        self.creation_date = creation_date
        self.owner = owner

    def __str__(self):
        return f'Данный медиа-файл имеет следующие параметры:\nИмя: {self.name}\nРазмер: {self.size} kb\nДата создания: {self.creation_date}\nВладелец: {self.owner}'

#делаем наследование

class Photo(MediaFiles):
    def file_type(self):
        print(f'Тип файла: Фото')

class Audio(MediaFiles):
    def file_type(self):
        print(f'Тип файла: Аудио')

class  Video(MediaFiles):
    def file_type(self):
        print(f'Тип файла: Видео')

class Shorts(Video, MediaFiles):
    def __init__(self, name, size, creation_date, owner, duration):
        super().__init__(name, size, creation_date, owner)
        self.duration = duration
    def __str__(self):
        return f'Данное видео имеет следующие параметры:\nИмя: {self.name}\nРазмер: {self.size} kb\nДата создания: {self.creation_date}\nВладелец: {self.owner}\nДлина видео: {self.duration} сек'
    def video_file_type(self):
        print(f'Тип файла: Видео, shorts') #имеются ввиду короткие видео определенного формата до 60 сек

class Long_Videos(Video, MediaFiles):
    def __init__(self, name, size, creation_date, owner, duration):
        super().__init__(name, size, creation_date, owner)
        self.duration = duration
    def __str__(self):
        return f'Данное видео имеет следующие параметры:\nИмя: {self.name}\nРазмер: {self.size} kb\nДата создания: {self.creation_date}\nВладелец: {self.owner}\nДлина видео: {self.duration} мин'
    def video_file_type(self):
        print(f'Тип файла: Видео, long') #ивидео больше 60 сек

class Text(MediaFiles):
    def file_type(self):
        print(f'Тип файла: Текст')

## пример действия:

target_file = Text('my_file',1044,'01-01-2024','Mikhail')
target_file = open('my_file.txt', 'w')
##разные действия с файлом
target_file.close()

