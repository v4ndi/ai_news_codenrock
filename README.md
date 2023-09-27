# Noname - AI_NEWS solution
Ноутбук с обучением можно найти `additional/FinTune_ruRoberta_large.ipynb`

# Запуск модели
Клонируйте репозиторий через git lfs clone

1. Зайти в папку `/src` 
2. Выполнить следующие команды при использовании видеокарты, используйте соотвествующие флаги
	`docker build -t ai_news .`
	`docker run -p 8000:8000 ai_news`
3. После запуска FastAPI интерфейса, нужно выполнить request запрос. Пример запроса лежит в `additional/request.py`. Входные данные в модель должны лежать в формате .csv со столбцами `[text, channel_id]` - пример входных данных можно найти `additional/for_testing.csv`. Файл  csv должен лежать в одной директории с  `request.py`
4. После выполнения request.py он вернет .csv файл `result_api_docker.csv` Который будет хранить уже готовый датасет с выставленными категориями и убранными дубликатами.
   Формат столбцов выходных данных:
   `[text, channel_id, category]`. Пример выходных данных `additional/result_api_docker.csv`

# Возможные проблемы 
Если возникнут проблемы выгрузки lfs файлов, то склонируйте репозиторий без них. И замените папку `/model` 
в директории `/src`. Выгрузить папку с файлами для модели можно по ссылке [ссылка](https://drive.google.com/drive/folders/1Wrryk9AaWzd6HyEMG9XB5sKkyOmDkfKk?usp=sharing) 

# Внимание
Категории записаны в конечном файле в следующес порядке, на id внимания не обращайте. Сверьте с правописанием категорий у вас
{0: 'Общее',
 1: 'Технологии',
 2: 'бизнес и стартап',
 3: 'блоги',
 4: 'видео и фильмы',
 5: 'дизайн',
 6: 'еда и кулинария',
 7: 'здоровье и медицина',
 8: 'игры',
 9: 'искусство',
 10: 'крипта',
 11: 'маркетинг',
 12: 'мода и красота',
 13: 'музыка',
 14: 'новости и сми',
 15: 'образование',
 16: 'политика',
 17: 'право',
 18: 'психология',
 19: 'путеш',
 20: 'развлечения',
 21: 'рукоделие',
 22: 'софт и приложения',
 23: 'спорт',
 24: 'финансы',
 25: 'фото',
 26: 'цитаты',
 27: 'шоу бизнес',
 28: 'экономика'}