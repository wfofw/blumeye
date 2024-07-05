import cv2
import numpy as np
import asyncio
import mss
import pyautogui
import os
import time
from pynput import keyboard
from concurrent.futures import ThreadPoolExecutor
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Переменная для управления остановкой программы
stop_program = False

# Создание пула потоков для параллельной обработки
executor = ThreadPoolExecutor(max_workers=50)

class StopProgramException(Exception):
    pass

# Функция для захвата экрана
def screen_capture(monitor):
    if stop_program:
        raise StopProgramException()
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray_image

# Функция для пороговой обработки
def apply_threshold(image):
    if stop_program:
        raise StopProgramException()
    _, thresh = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
    return thresh

# Функция для нахождения и упрощения контуров
def find_and_simplify_contours(image):
    if stop_program:
        raise StopProgramException()
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    epsilon = 0.01 * max(cv2.arcLength(cnt, True) for cnt in contours) if contours else 0
    simplified_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    return simplified_contours

# Функция для фильтрации прямых контуров
def filter_straight_contours(contours, min_aspect_ratio=5):
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0 or max(w / h, h / w) > min_aspect_ratio:
            continue
        filtered_contours.append(contour)
    return filtered_contours

# Функция для клика по найденной области
def click_on_contour(contour, monitor):
    if stop_program:
        raise StopProgramException()
    x, y, w, h = cv2.boundingRect(contour)
    center_x = monitor['left'] + x + w / random.uniform(1.210, 1.999)
    center_y = monitor['top'] + y + h / random.uniform(1.210, 1.999)
    pyautogui.click(center_x, center_y)

# Функция для сохранения скриншотов
def save_screenshot(image, directory='screenshots', prefix='screenshot'):
    if stop_program:
        raise StopProgramException()
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{directory}/{prefix}_{timestamp}.png"
    cv2.imwrite(filename, image)

# Функция для визуализации контуров
def visualize_contours(image, contours, matched_contour=None):
    if stop_program:
        raise StopProgramException()
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)  # Зеленые контуры
    if matched_contour is not None:
        cv2.drawContours(vis_image, [matched_contour], -1, (0, 0, 255), 2)  # Красный контур
    return vis_image

# Обработчик нажатия клавиш для остановки программы
def on_press(key):
    global stop_program
    if hasattr(key, 'char') and key.char == 'q':
        stop_program = True
        return False  # Остановить слушатель

# Функция для обработки скриншота
def process_screenshot(monitor):
    if stop_program:
        return
    try:
        capture_image, gray_image = screen_capture(monitor)
        capture_thresh = apply_threshold(gray_image)
        contours = find_and_simplify_contours(capture_thresh)
        if contours:
            contours = filter_straight_contours(contours)  # Фильтрация прямых контуров
            for contour in contours:
                if 305 < cv2.contourArea(contour) < 10000:
                    click_on_contour(contour, monitor)
                    vis_image = visualize_contours(capture_thresh, contours, contour)
                    save_screenshot(vis_image, prefix='matched')
                    break
    except StopProgramException:
        return

# Функция для клика по определенной точке
def click_fixed_point(x, y):
    pyautogui.click(x, y)

# Функция для рандомного клика
async def random_click_task():
    while not stop_program:
        await asyncio.sleep(random.uniform(1.0001, 3.9992))
        fixed_click_point1 = (random.randint(387, 390) + random.randint(9, 170), random.randint(1019, 1025) + random.randint(1, 6))
        fixed_click_point2 = (random.randint(1406, 1446) + random.randint(7, 50), random.randint(1019, 1025) + random.randint(1, 6))

        random_fixed_click_point = random.choice([fixed_click_point1, fixed_click_point2])
        click_fixed_point(*random_fixed_click_point)

# Главная асинхронная функция
async def main():
    global stop_program

    # Определение рабочего окна (монитора)
    monitor = {
        "top": 90,    # Верхняя координата окна
        "left": 562,   # Левая координата окна
        "width": 767,  # Ширина окна (замените на ширину вашего экрана)
        "height": 920  # Высота окна (замените на высоту вашего экрана)
    }

    # Запуск слушателя для нажатий клавиш
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    futures = []

    # Запуск задачи для рандомного клика
    asyncio.create_task(random_click_task())

    while not stop_program:
        # Запуск задачи в отдельном потоке
        future = executor.submit(process_screenshot, monitor)
        futures.append(future)

        await asyncio.sleep(0.041)

    # Принудительное завершение всех запущенных задач
    for future in futures:
        future.cancel()

    executor.shutdown(wait=False)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())
