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
executor = ThreadPoolExecutor(max_workers=38)

# Функция для захвата экрана
async def screen_capture(monitor):
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_image

# Функция для пороговой обработки
def apply_threshold(image):
    _, thresh = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)
    return thresh

# Функция для нахождения и упрощения контуров
def find_and_simplify_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    epsilon = 0.01 * max(cv2.arcLength(cnt, True) for cnt in contours) if contours else 0
    simplified_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    return simplified_contours

# Функция для клика по найденной области
def click_on_contour(contour, monitor):
    x, y, w, h = cv2.boundingRect(contour)
    center_x = monitor['left'] + x + w // 2
    center_y = monitor['top'] + y + h // 2
    pyautogui.click(center_x, center_y)

# Функция для сохранения скриншотов
def save_screenshot(image, directory='screenshots', prefix='screenshot'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{directory}/{prefix}_{timestamp}.png"
    cv2.imwrite(filename, image)

# Функция для визуализации контуров
def visualize_contours(image, contours, matched_contour=None):
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

# Функция для клика по определенной точке
def click_fixed_point(x, y):
    pyautogui.click(x, y)

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

    while not stop_program:
        fixed_click_point1 = (random.randint(387, 390) + random.randint(9, 174) // 2, random.randint(1019, 1025) + random.randint(1, 6) // 2)
        fixed_click_point2 = (random.randint(1406, 1446) + random.randint(7, 50) // 2, random.randint(1019, 1025) + random.randint(1, 6) // 2)

        random_click_left_side = (random.randint(11, 301) + random.randint(1, 10) // 2, random.randint(320, 750) + random.randint(11, 50) // 2) #random
        random_click_right_side = (random.randint(1580, 1800) + random.randint(1, 12) // 2, random.randint(320, 750) + random.randint(11, 50) // 2) #random

        random_fixed_click_point = random.choice([
            fixed_click_point1,
            random_click_left_side,
            fixed_click_point2,
            random_click_right_side
        ])
        # Захват экрана и обработка изображения
        capture_image = await screen_capture(monitor)
        capture_thresh = apply_threshold(capture_image)
        contours = find_and_simplify_contours(capture_thresh)
        if contours is None:
            continue

        if contours:
            click_on_contour(contours[0], monitor)
            # Сохранение скриншота с найденным контуром
            vis_image = visualize_contours(capture_thresh, contours, contours[0])
            save_screenshot(vis_image, prefix='matched')
        else:
            await asyncio.sleep(random.uniform(0.62, 1.0))
            click_fixed_point(*random_fixed_click_point)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())
