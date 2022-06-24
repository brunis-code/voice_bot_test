from sklearn.feature_extraction.text import TfidfVectorizer  # для векторизации текста
from sklearn.linear_model import LogisticRegression  # для классификации намерений
import zipfile  # для распаковки архива датасета с диалогами
import os.path  # для проверки наличия файла
import random  # для генерации случайных ответов
import nltk  # библиотека для естественной обработки языка
import json  # представление в качестве JSON

# векторизатор текста
vectorizer = None

# классификатор запросов
classifier = None

# датасет на основе открытых диалогов
dataset = {}  # {слово: [[запрос, ответ], [запрос 2, ответ 2], ...], ...}

# порог вероятности, при котором на намерение пользователя будет отправляться ответ из bot_config
threshold = 0.7

# ведение статистики ответов
stats = {"intent": 0, "generative": 0, "failure": 0}

# конфигурация бота с намерениями действия, примерами запросов и ответов на них
# ниже продемонстрирован способ загрузки из файла
bot_config = {
    "intents": {
        "hello": {
            "examples": ["Привет", "Здравствуйте", "Добрый день"],
            "responses": ["Привет", "Здравствуй", "Предлагаю сразу к делу :)"]
        },
        "bye": {
            "examples": ["Пока", "До свидания", "Увидимся"],
            "responses": ["Пока", "Веди себя хорошо"]
        },
    },

    "failure_phrases": [
        "Не знаю, что сказать даже",
        "Меня не научили отвечать на такое",
        "Я не знаю, как отвечать на такое"
    ]
}

"""
Иинициализация бота
"""

# можно загрузить конфиг из файла, если он большой
with open("bot_corpus/bot_config.json", encoding="utf-8") as file:
    bot_config = json.load(file)



def get_bot_response(request: str):
    """
    Отправка ответа пользователю на его запрос с учётом статистики
    :param request: запрос пользователя
    :return: ответ для пользователя
    """
    # определение намерения пользователя,
    # использование заготовленного ответа
    intent = get_intent(request)
    if intent:
        stats["intent"] += 1
        return get_response_by_intent(intent)

    # если нет заготовленного ответа - идёт поиск ответа в датасете диалогов
    response = get_generative_response(request)
    if response:
        stats["generative"] += 1
        return response

    # если бот не может подобрать ответ - отправляется ответ-заглушка
    stats["failure"] += 1
    return get_failure_phrase()

def get_intent(request: str):
    """
    Получение наиболее вероятного намерения пользователя из сообщения
    :param request: запрос пользователя
    :return: наилучшее совпадение
    """
    question_probabilities = classifier.predict_proba(vectorizer.transform([request]))[0]
    best_intent_probability = max(question_probabilities)

    if best_intent_probability > threshold:
        best_intent_index = list(question_probabilities).index(best_intent_probability)
        best_intent = classifier.classes_[best_intent_index]
        return best_intent

    return None

def get_response_by_intent(intent: str):
    """
    Получение случайного ответа на намерение пользователя
    :param intent: намерение пользователя
    :return: случайный ответ из прописанных для намерения
    """
    phrases = bot_config["intents"][intent]["responses"]
    return random.choice(phrases)

def normalize_request(request):
    """
    Приведение запроса пользователя к нормальному виду путём избавления от лишних символов и смены регистра
    :param request: запрос пользователя
    :return: запрос пользователя в нижнем регистре без спец-символов
    """
    normalized_request = request.lower().strip()
    alphabet = " -1234567890йцукенгшщзхъфывапролджэёячсмитьбю"
    normalized_request = "".join(character for character in normalized_request if character in alphabet)
    return normalized_request

def get_generative_response(request: str):
    """
    Подбор ответа, получаемого из открытого датасета диалогов на основе поиска максимального совпадения
    :param request: запрос пользователя
    :return: ответ из датасета лиалогов
    """
    phrase = normalize_request(request)
    words = phrase.split(" ")

    mini_dataset = []
    for word in words:
        if word in dataset:
            mini_dataset += dataset[word]

    candidates = []

    for question, answer in mini_dataset:
        if abs(len(question) - len(request)) / len(question) < 0.4:
            distance = nltk.edit_distance(question, request)
            score = distance / len(question)
            if score < 0.4:
                candidates.append([question, answer, score])

    if candidates:
        return min(candidates, key=lambda candidate: candidate[0])[1]

    return None

def get_failure_phrase():
    """
    Если бот не может ничего ответить - будет отправлена случайная фраза из списка failure_phrases в bot_config
    :return: случайная фраза в случае провала подбора ответа ботом
    """
    phrases = bot_config["failure_phrases"]
    return random.choice(phrases)

def create_bot_config_corpus():
    """
    Создание и обучение корпуса для бота, обученного на bot_config для дальнейшей обработки запросов пользователя
    """
    corpus = []
    y = []

    for intent, intent_data in bot_config["intents"].items():
        for example in intent_data["examples"]:
            corpus.append(example)
            y.append(intent)

    # векторизация
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    x = vectorizer.fit_transform(corpus)

    # классификация
    classifier = LogisticRegression()
    classifier.fit(x, y)

    print("Обучение на файле конфигурации завершено")

def create_bot_dialog_dataset():
    """
    Загрузка датасета диалогов для чат-бота путём парсинга файла
    Открытые датасеты диалогов для обучения бота: https://github.com/Koziev/NLP_Datasets
    Можно использовать выгрузку истории сообщений из собственных диалогов ВКонтакте в таком же виде
    """

    if not os.path.isfile("bot_corpus/dialogues.txt"):
        with zipfile.ZipFile("bot_corpus/dialogues.zip", "r") as zip_file:
            zip_file.extractall("bot_corpus")
            print("Распаковка датасета завершена")

    with open("bot_corpus/dialogues.txt", encoding="utf-8") as file:
        content = file.read()

    dialogues = content.split("\n\n")
    questions = set()

    for dialogue in dialogues:
        phrases = dialogue.split("\n")[:2]
        if len(phrases) == 2:
            question, answer = phrases
            question = normalize_request(question[2:])
            answer = answer[2:]

            if question and question not in questions:
                questions.add(question)
                words = question.split(" ")
                for word in words:
                    if word not in dataset:
                        dataset[word] = []
                    dataset[word].append([question, answer])

    too_popular = set()
    for word in dataset:
        if len(dataset[word]) > 10000:
            too_popular.add(word)

    for word in too_popular:
        dataset.pop(word)

    print("Загрузка датасета диалогов завершена")
# обучение бота и подготовка датасета с готовыми диалогами
create_bot_config_corpus()
create_bot_dialog_dataset()
