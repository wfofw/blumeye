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

# Пороговое значение для исключения бомб
BOMB_THRESHOLD = 0.00005  # Уменьшено пороговое значение для более строгой фильтрации
HUMOMENTS_THRESHOLD = 0.0000000001  # Строгий порог для HuMoments

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
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    return thresh

# Функция для нахождения и упрощения контуров
def find_and_simplify_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    epsilon = 0.01 * max(cv2.arcLength(cnt, True) for cnt in contours) if contours else 0
    simplified_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    return simplified_contours

# Функция для сопоставления контуров с дополнительной фильтрацией
def match_contours(template_contours, contours, threshold=0.1):
    for template_contour in template_contours:
        for contour in contours:
            if 100 < cv2.contourArea(contour) < 5000:  # Фильтрация по размеру контура
                match = cv2.matchShapes(template_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
                if match < threshold:
                    return contour
    return None

# Функция для исключения бомб с использованием HuMoments и cv2.matchShapes
def is_bomb(contour, bomb_contours):
    for bomb_contour in bomb_contours:
        match = cv2.matchShapes(contour, bomb_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        if match < BOMB_THRESHOLD:
            hu_moments_contour = cv2.HuMoments(cv2.moments(contour)).flatten()
            hu_moments_bomb = cv2.HuMoments(cv2.moments(bomb_contour)).flatten()
            hu_diff = np.sum(np.abs(hu_moments_contour - hu_moments_bomb))
            if hu_diff < HUMOMENTS_THRESHOLD:  # Ещё более строгий порог для HuMoments
                print(f"Bomb detected with match value: {match}, hu_diff: {hu_diff}")  # Отладочное сообщение
                return True
    return False

# Функция для клика по найденной области
def click_on_contour(contour, monitor):
    x, y, w, h = cv2.boundingRect(contour)
    center_x = monitor['left'] + x + w / random.uniform(1.210, 1.999)
    center_y = monitor['top'] + y + h / random.uniform(1.210, 1.999)
    pyautogui.click(center_x, center_y)

# Функция для сохранения скриншотов
def save_screenshot(image, directory='screenshots', prefix='screenshot'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{directory}/{prefix}_{timestamp}.png"
    cv2.imwrite(filename, image)

# Функция для визуализации контуров
def visualize_contours(image, contours, matched_contour, bomb_contours):
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)  # Зеленые контуры
    for bomb_contour in bomb_contours:
        cv2.drawContours(vis_image, [bomb_contour], -1, (255, 0, 0), 2)  # Синие контуры - бомбы
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

    # Загрузка эталонных изображений бомб и нахождение их контуров
    bomb_dir = './bombEtalon/'
    bomb_contours = []
    for file_name in os.listdir(bomb_dir):
        bomb_image = cv2.imread(os.path.join(bomb_dir, file_name), cv2.IMREAD_GRAYSCALE)
        bomb_thresh = apply_threshold(bomb_image)
        contours = find_and_simplify_contours(bomb_thresh)
        if contours:
            bomb_contours.append(contours[0])

    if not bomb_contours:
        print("Ошибка: не удалось найти контуры бомб.")
        return

    # Загрузка эталонных изображений и нахождение их контуров
    template_dir = './blumEtalon/'
    template_contours = []
    for file_name in os.listdir(template_dir):
        template_image = cv2.imread(os.path.join(template_dir, file_name), cv2.IMREAD_GRAYSCALE)
        template_thresh = apply_threshold(template_image)
        contours = find_and_simplify_contours(template_thresh)
        if contours:
            template_contours.append(contours[0])

    if not template_contours:
        print("Ошибка: не удалось найти контуры в эталонных изображениях.")
        return

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
        fixed_click_point1 = (random.randint(387, 390) + random.randint(9, 174) // 2, random.randint(912, 950) + random.randint(20, 99) // 2)
        fixed_click_point2 = (random.randint(1406, 1446) + random.randint(7, 96) // 2, random.randint(910, 948) + random.randint(23, 96) // 2)

        random_click_left_side = (random.randint(11, 301) + random.randint(1, 10) // 2, random.randint(302, 489) + random.randint(9, 83) // 2) #random
        random_click_right_side = (random.randint(1599, 1800) + random.randint(1, 12) // 2, random.randint(298, 470) + random.randint(13, 88) // 2) #random

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

        # Параллельное сопоставление контуров
        loop = asyncio.get_running_loop()
        matched_contour = await loop.run_in_executor(executor, match_contours, template_contours, contours)
        if matched_contour is not None:
            bomb_check = await loop.run_in_executor(executor, is_bomb, matched_contour, bomb_contours)
            if not bomb_check:
                click_on_contour(matched_contour, monitor)
                # Сохранение скриншота с найденным контуром
                vis_image = visualize_contours(capture_thresh, contours, matched_contour, bomb_contours)
                save_screenshot(vis_image, prefix='matched')
            else:
                # Сохранение скриншота без совпадений
                vis_image = visualize_contours(capture_thresh, contours, None, bomb_contours)
                save_screenshot(vis_image, prefix='screen')
        else:
            await asyncio.sleep(random.uniform(0.42, 1.0))
            click_fixed_point(*random_fixed_click_point)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())
