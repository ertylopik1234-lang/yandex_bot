import time
import random
import pickle
import os
import base64
import requests
from io import BytesIO
from PIL import Image
from flask import Flask
import threading
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration

COOKIE_B64 = os.environ.get("COOKIES_B64")
HEADLESS_MODE = True
TASK_DURATION_HOURS = 24
MIN_PAUSE = 180
MAX_PAUSE = 300
NO_TASKS_WAIT = 1800

print("⏳ Загрузка текстового ИИ...")
generator = pipeline('text-generation', model='microsoft/DialoGPT-medium')
print("✅ Текстовый ИИ загружен")

print("⏳ Загрузка модели для картинок...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
image_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("✅ Модель для картинок загружена")

def ask_ai(prompt):
    full = f"Вопрос: {prompt}\nОтвет:"
    result = generator(full, max_length=50, num_return_sequences=1)
    return result[0]['generated_text'].replace(full, "").strip() or "Да"

def ask_ai_for_choice(question, options):
    prompt = f"Вопрос: {question}\nВарианты: {', '.join(options)}\nПравильный вариант (только текст):"
    result = generator(prompt, max_length=30, num_return_sequences=1)
    answer = result[0]['generated_text'].replace(prompt, "").strip()
    for opt in options:
        if opt.lower() in answer.lower():
            return opt
    return options[0]

def analyze_image_from_element(img_element):
    try:
        src = img_element.get_attribute("src")
        if not src:
            return "Изображение не найдено"
        response = requests.get(src, timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        out = image_model.generate(**inputs, max_length=50)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"Ошибка: {e}"

def choose_option_by_image_description(description, labels):
    prompt = f"Описание картинки: {description}\nВарианты: {', '.join(labels)}\nКакой вариант соответствует картинке? Только текст:"
    result = generator(prompt, max_length=30, num_return_sequences=1)
    answer = result[0]['generated_text'].replace(prompt, "").strip()
    for opt in labels:
        if opt.lower() in answer.lower():
            return opt
    return labels[0]

def is_field_task(driver):
    try:
        body = driver.find_element(By.TAG_NAME, "body").text.lower()
        if any(k in body for k in ["геолокация", "камера", "фото"]):
            return True
        if driver.find_elements(By.XPATH, "//input[@type='file']"):
            return True
        return False
    except:
        return False

def analyze_task_type(driver):
    return {
        "has_text": bool(driver.find_elements(By.TAG_NAME, "textarea")),
        "has_choice": bool(driver.find_elements(By.XPATH, "//input[@type='radio']") or
                          driver.find_elements(By.XPATH, "//input[@type='checkbox']"))
    }

def load_cookies_secure(driver):
    if not COOKIE_B64:
        print("❌ Переменная COOKIES_B64 не найдена")
        return False
    try:
        data = base64.b64decode(COOKIE_B64)
        cookies = pickle.loads(data)
        for cookie in cookies:
            driver.add_cookie(cookie)
        return True
    except Exception as e:
        print(f"❌ Ошибка загрузки кук: {e}")
        return False

def has_available_tasks(driver) -> bool:
    try:
        start_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Начать')]")
        if start_buttons:
            return True
        no_tasks_text = driver.find_elements(By.XPATH, "//*[contains(text(), 'Нет заданий')]")
        if no_tasks_text:
            return False
        return False
    except:
        return False

def perform_task(driver) -> bool:
    driver.get("https://yandex.ru/tasks")
    time.sleep(random.uniform(1.5, 3))

    try:
        start_btn = WebDriverWait(driver, 12).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Начать')]"))
        )
        start_btn.click()
    except:
        return False

    time.sleep(random.uniform(1, 3))

    if is_field_task(driver):
        print("⏭️ Полевое пропущено")
        driver.get("https://yandex.ru/tasks")
        return False

    info = analyze_task_type(driver)
    print(f"🔍 Текст: {info['has_text']}, Выбор: {info['has_choice']}")

    img_elements = driver.find_elements(By.TAG_NAME, "img")
    image_description = None
    if img_elements:
        print("🖼️ Анализируем картинку...")
        image_description = analyze_image_from_element(img_elements[0])
        print(f"📝 Описание: {image_description}")

    if info["has_text"]:
        for inp in driver.find_elements(By.TAG_NAME, "textarea"):
            if inp.is_displayed():
                placeholder = inp.get_attribute("placeholder") or "Ответьте на вопрос"
                if image_description:
                    answer = ask_ai(f"{placeholder} (на картинке: {image_description})")
                else:
                    answer = ask_ai(placeholder)
                inp.send_keys(answer)
                time.sleep(0.5)

    if info["has_choice"]:
        labels = driver.find_elements(By.XPATH, "//label")
        options = [lbl.text.strip() for lbl in labels if lbl.text.strip()]
        if options:
            question = driver.find_element(By.TAG_NAME, "h2").text if driver.find_elements(By.TAG_NAME, "h2") else "Выберите вариант"
            if image_description:
                chosen = choose_option_by_image_description(image_description, options)
            else:
                chosen = ask_ai_for_choice(question, options)
            print(f"🤖 ИИ выбрал: {chosen}")
            for lbl in labels:
                if chosen.lower() in lbl.text.lower():
                    lbl.click()
                    break
        radios = driver.find_elements(By.XPATH, "//input[@type='radio']")
        if radios:
            radios[0].click()
        checks = driver.find_elements(By.XPATH, "//input[@type='checkbox']")
        if checks:
            checks[0].click()

    try:
        submit_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Отправить')]")
        submit_btn.click()
    except:
        try:
            submit_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Готово')]")
            submit_btn.click()
        except:
            return False

    time.sleep(random.uniform(1.5, 3))
    return True

def run_flask():
    app = Flask('')
    @app.route('/')
    def home():
        return "Bot is running"
    app.run(host='0.0.0.0', port=10000)

def main():
    threading.Thread(target=run_flask, daemon=True).start()
    
    options = Options()
    if HEADLESS_MODE:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(options=options)

    driver.get("https://yandex.ru/tasks")
    time.sleep(3)

    if load_cookies_secure(driver):
        driver.refresh()
        time.sleep(2)
        print("✅ Авторизация по кукам (Base64)")
    else:
        print("⚠️ Ошибка авторизации. Проверьте переменную COOKIES_B64")
        driver.quit()
        return

    print("🚀 Начинаем выполнение заданий...")
    start_time = time.time()
    done = 0

    while time.time() - start_time < 3600 * TASK_DURATION_HOURS:
        try:
            driver.get("https://yandex.ru/tasks")
            time.sleep(3)

            if not has_available_tasks(driver):
                print("😴 Нет доступных заданий. Ожидание 30 минут...")
                time.sleep(NO_TASKS_WAIT)
                continue

            if perform_task(driver):
                done += 1
                print(f"✅ Выполнено: {done}")
            else:
                print("⚠️ Задание не выполнено")

            wait = random.randint(MIN_PAUSE, MAX_PAUSE)
            print(f"⏳ Пауза {wait//60} мин {wait%60} сек")
            time.sleep(wait)

        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            time.sleep(60)

    driver.quit()
    print(f"🎯 За {TASK_DURATION_HOURS} час(ов) выполнено {done} заданий")

if __name__ == "__main__":
    main()
