import asyncio
import json
import random
import re
import requests
import sounddevice as sd
import pyttsx3
import webbrowser
from vosk import Model, KaldiRecognizer
import numpy as np

# Константы для настроек речи ассистента и считывание звука микрофона
SAMPLE_RATE = 44100
BLOCK_SIZE = 8000
SPEECH_RATE = 400
MIN_SPEECH_GAP = 1.5
AMPLITUDE_THRESHOLD = 0
MAX_CHARACTER_ID = 826

#Словарь перевода цифр для получения id
RU_NUMBER_WORDS = {
    "ноль": "0",
    "один": "1", "одна": "1",
    "два": "2",
    "три": "3",
    "четыре": "4",
    "пять": "5",
    "шесть": "6",
    "семь": "7",
    "восемь": "8",
    "девять": "9",
}

# Фразы ассистента (огурчика рика) для озвучки
Answers = {
    "welcome": [
        "Огурчик Рик на связи! Не трать мое время, говори быстро! Для начала, скажи 'персонаж ID' или 'случайный персонаж'."
    ],
    "processing": [
        "Обрабатываю... Не отвлекай гениального Огурчика!"
    ],
    "success": [
        "Вот тебе: {result}. Благодари Огурчика Рика!"
    ],
    "failure": [
        "Не трынди ерунду! Скажи нормально, или вали!"
    ],
    "shutdown": [
        "Огурчик Рик уходит в закат. Пока, лузеры!"
    ],
    "no_character_selected": [
        "Эй, сначала персонажа выбери! Скажи 'персонаж ID' или 'случайный персонаж', чтобы я знал, с кем возиться!",
    ],
    "character_changed_prompt": [
        "Ладно, этого забыли. Кого следующего будем мучать? Говори ID или 'случайный персонаж'."
    ],
    "character_selection_success": [
        "Персонаж {name} выбран. Не заставляй меня ждать."
    ]
}


class PickleRickAssistant:
    def __init__(self, sample_rate):
        try:
            self.model = Model('vosk-model-small-ru-0.22')  # Модель распознавания речи
            self.recognizer = KaldiRecognizer(self.model, sample_rate) #Распознаватель
            self.recognizer.SetMaxAlternatives(0)
            self.recognizer.SetWords(True)
            self.speech_engine = pyttsx3.init() #Синтез речи для ответов ассистента
            self.speech_engine.setProperty("rate", SPEECH_RATE)
            self.last_spoken = 0.0
            self.character_data = {}
            self.current_character_id = None
        except Exception as e:
            print(f"❌ Ошибка при инициализации Vosk: {e}")
            raise
    #Функция для озвучивания ответов ассистента с полученным результатом запроса
    def speak(self, quote_type, result=None):
        try:
            quote = random.choice(Answers[quote_type])
            if result:
                if isinstance(result, dict):
                    quote = quote.format(**result)
                else:
                    quote = quote.format(result=result)
            print(f"🥒 Огурчик Рик: {quote}")
            self.speech_engine.say(quote)
            self.speech_engine.runAndWait()
            self.last_spoken = asyncio.get_event_loop().time()
        except Exception as e:
            print(f"❌ Ошибка при воспроизведении речи: {e}")
    #Получение ответа запроса с заданным ID персонажа
    async def fetch_character_by_id(self, char_id):
        try:
            self.speak("processing")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                requests.get,
                f"https://rickandmortyapi.com/api/character/{char_id}"
            )
            #Обработка ошибки статуса запроса
            if response.status_code != 200:
                self.current_character_id = None
                self.character_data = {}
                self.speak("failure", result=f"Персонаж с ID {char_id} не найден. Попробуй другой номер, гений.")
                return

            self.character_data = response.json()
            self.current_character_id = self.character_data["id"]
            self.speak("character_selection_success", result={"name": self.character_data['name']})
        except Exception as e:
            print(f"❌ Ошибка при получении персонажа по ID: {e}")
            self.current_character_id = None
            self.character_data = {}
            self.speak("failure")
    #Аналогичная функция для случайного ID
    async def fetch_random_character(self):
        try:
            self.speak("processing")
            char_id_random = random.randint(1, MAX_CHARACTER_ID)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                requests.get,
                f"https://rickandmortyapi.com/api/character/{char_id_random}"
            )

            if response.status_code != 200:
                self.current_character_id = None
                self.character_data = {}
                self.speak("failure", result="Не смог вытащить случайного. API глючит, или я.")
                return

            self.character_data = response.json()
            self.current_character_id = self.character_data["id"]
            self.speak("character_selection_success", result={"name": self.character_data['name']})
        except Exception as e:
            print(f"❌ Ошибка при получении случайного персонажа: {e}")
            self.current_character_id = None
            self.character_data = {}
            self.speak("failure")
    #Функция смены ID персонажа
    async def change_character(self):
        self.current_character_id = None
        self.character_data = {}
        self.speak("character_changed_prompt")
    #Функция поиска первого эпизода появления персонажа
    async def get_first_episode(self):
        #Обработка ошибки незаданного ID
        if not self.current_character_id:
            self.speak("no_character_selected")
            return
        try:
            self.speak("processing")
            episode_url = self.character_data["episode"][0]
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, requests.get, episode_url)
            episode_data = response.json()
            self.speak("success", result=episode_data["name"])
        except Exception as e:
            print(f"❌ Ошибка при получении эпизода: {e}")
            self.speak("failure")
    #Функция показа изображения персонажа
    async def display_image(self):
        #Обработка ошибки незаданного ID
        if not self.current_character_id:
            self.speak("no_character_selected")
            return
        try:
            self.speak("processing")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, webbrowser.open, self.character_data["image"])
            self.speak("success", result="Вот тебе картинка, наслаждайся!")
        except Exception as e:
            print(f"❌ Ошибка при открытии изображения: {e}")
            self.speak("failure")
    #Функция получения вида/расы персонажа
    async def get_species(self):
        #Обработка ошибки незаданного ID
        if not self.current_character_id:
            self.speak("no_character_selected")
            return
        try:
            self.speak("processing")
            species = self.character_data.get("species", "неизвестный вид")
            self.speak("success", result=f"Вид — {species}")
        except Exception as e:
            print(f"❌ Ошибка при получении вида: {e}")
            self.speak("failure")
    #Функция получения гендера персонажа
    async def get_gender(self):
        #Обработка ошибки незаданного ID
        if not self.current_character_id:
            self.speak("no_character_selected")
            return
        try:
            self.speak("processing")
            gender = self.character_data.get("gender", "неизвестный пол")
            self.speak("success", result=f"Пол — {gender}")
        except Exception as e:
            print(f"❌ Ошибка при получении пола: {e}")
            self.speak("failure")
    #Краткая сводка информации выбранного персонажа
    async def get_current_character_info(self):
        if not self.current_character_id:
            self.speak("no_character_selected")
            return
        if self.character_data and "name" in self.character_data:
            info = (f"Имя: {self.character_data['name']}, "
                    f"Статус: {self.character_data['status']}, "
                    f"Вид: {self.character_data['species']}, "
                    f"Гендер: {self.character_data['gender']}.")
            self.speak("success",
                       result=f"Сейчас работаем с {self.character_data['name']}. Вот что я о нем знаю: {info}")
        else:
            self.speak("failure", result="Я сам не помню, с кем я. Выбери кого-нибудь еще раз!")
    #Обработка голосовых команд
    async def process_command(self, text):
        command = text.lower().strip()
        if not command:
            return True
        print(f"👤 Пользователь: {command}")

        #Блок получения ID
        match_spoken_digits = re.match(r"(?:персонаж|выбери персонажа|выбрать персонажа)\s+(?:номер\s+)?(.*)", command)
        if match_spoken_digits:
            words_after_command = match_spoken_digits.group(1).strip()

            if words_after_command:
                potential_digit_words = words_after_command.split()
                id_str_from_words = ""

                for word in potential_digit_words:
                    cleaned_word = word.rstrip(".?!,:;")

                    if cleaned_word in RU_NUMBER_WORDS:
                        id_str_from_words += RU_NUMBER_WORDS[cleaned_word]
                    else:
                        break

                if id_str_from_words:
                    try:
                        char_id = int(id_str_from_words)
                        if 1 <= char_id <= MAX_CHARACTER_ID:
                            await self.fetch_character_by_id(char_id)
                        else:
                            self.speak("failure",
                                       result=f"Собранный ID: {char_id} (из слов '{words_after_command}') вне диапазона от 1 до {MAX_CHARACTER_ID}.")
                        return True
                    except ValueError:
                        self.speak("failure",
                                   result=f"Не удалось преобразовать '{id_str_from_words}' в число из '{words_after_command}'.")
                        return True

        #Обработка вышеописанных команд
        if any(keyword in command for keyword in ["случайный персонаж", "рандомный персонаж", "дай случайного"]):
            await self.fetch_random_character()
            return True

        if any(keyword in command for keyword in
               ["смени персонажа", "смена персонажа", "другой персонаж", "выбрать другого"]):
            await self.change_character()
            return True


        if any(keyword in command for keyword in ["выход", "стоп", "завершить"]):  
            self.speak("shutdown")
            return False


        if not self.current_character_id:
            self.speak("no_character_selected")
            return True

        if any(word in command for word in ["эпизод", "серия"]):
            await self.get_first_episode()
        elif any(word in command for word in ["покажи", "изображение", "картинка"]):
            await self.display_image()
        elif any(word in command for word in ["вид", "раса", "тип"]):
            await self.get_species()
        elif any(word in command for word in ["гендер", "пол"]):
            await self.get_gender()
        elif any(word in command for word in
                 ["кто ты", "какой персонаж", "текущий персонаж", "с кем работаешь", "информация о персонаже", "инфо"]):
            await self.get_current_character_info()
        else:
            self.speak("failure")
        return True

#Функция аудиозахвата ввода
async def audio_listener(assistant, device_index, sample_rate):
    try:
        #Информация о выбранном устройстве захвата
        print(f"🎙️ Используется устройство: {sd.query_devices(device_index)['name']}")
        device_info = sd.query_devices(device_index)
        print(f"🎙️ Поддерживаемая частота устройства: {device_info['default_samplerate']} Гц")
        print(f"🎙️ Формат: моно, 16 бит, {sample_rate} Гц")
        with sd.RawInputStream(
                samplerate=sample_rate, #Частота дискретизации
                blocksize=BLOCK_SIZE,
                device=device_index,
                dtype="int16",
                channels=1, #Моно канал
        ) as stream:
            print("🎙️ Слушаю...")
            while True:
                data, overflowed = stream.read(BLOCK_SIZE) #overflowed булево значение
                if overflowed:
                    #Обработка переполнения буфера
                    print("⚠️ Переполнение аудиобуфера!")
                    continue
                #Обрабатывается только достаточно амплитудные данные (если ввод слишком тихий - шум)
                amplitude = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
                if amplitude < AMPLITUDE_THRESHOLD:
                    continue

                data_bytes = np.frombuffer(data, dtype=np.int16).tobytes()
                #Передача данных в распознаватель
                if assistant.recognizer.AcceptWaveform(data_bytes):
                    result = json.loads(assistant.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(f"🔊 Распознан текст: {text}")
                        if (asyncio.get_event_loop().time() - assistant.last_spoken) >= MIN_SPEECH_GAP:
                            if not await assistant.process_command(text):
                                break
                else:
                    partial_result = json.loads(assistant.recognizer.PartialResult())
                    partial_text = partial_result.get("partial", "").strip()
                    if partial_text:
                        print(f"🔊 Частичное распознавание: {partial_text}")
                await asyncio.sleep(0.01)
    except Exception as e:
        print(f"❌ Ошибка в audio_listener: {e}")
        raise

#Основной блок логики кода
async def main():
    device_index = 3 #Индекс микрофона
    sample_rate = SAMPLE_RATE

    try:
        device_info = sd.query_devices(device_index)
        if sample_rate != device_info['default_samplerate']:
            print(
                f"⚠️ Заданная частота {sample_rate} Гц отличается от дефолтной для устройства ({device_info['default_samplerate']} Гц). Используем {sample_rate}.")
    except Exception as e:
        print(f"❌ Не удалось получить информацию об устройстве {device_index}: {e}")
        print("Пожалуйста, проверьте индекс устройства и доступность микрофона.")
        return

    try:
        assistant = PickleRickAssistant(sample_rate)
        assistant.speak("welcome")
        await audio_listener(assistant, device_index, sample_rate)
    except Exception as e:
        print(f"❌ Ошибка при инициализации или работе ассистента: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🚪 Программа завершена пользователем.")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")