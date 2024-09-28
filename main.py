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
executor = ThreadPoolExecutor(max_workers=400)

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
            if 450 < cv2.contourArea(contour) < 5000:  # Фильтрация по размеру контура
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
    center_x = monitor['left'] + x + w / random.uniform(1.000001, 1.999999)
    center_y = monitor['top'] + y + h / random.uniform(1.000001, 1.999999)
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
    stop_program = True
    return False  # Остановить слушатель и закончить программу

async def stop_listener():
    # Запускаем слушателя клавиатуры в фоновом режиме
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    while not stop_program:
        await asyncio.sleep(0.1)  # Легкая пауза для фона, чтобы не блокировать основной цикл
    listener.stop()

# Функция для клика по определенной точке
def click_fixed_point(x, y):
    pyautogui.click(x, y)

def back_to_menu():
    pyautogui.click(100, 130)
    time.sleep(5)

def start_button():
    pyautogui.moveTo(1700, 800)
    pyautogui.mouseDown()
    time.sleep(0.5)
    pyautogui.moveTo(1600, 0)
    pyautogui.mouseUp

async def perform_intermediate_task():
    if not stop_program:
        print("Ебашим в лобби и запуск")
        back_to_menu()
        start_button()
        start_button()
        start_button()

async def run_main_for_duration(main_task_duration, pause_duration):
    global stop_program

    while not stop_program:
        # Запуск main() на 45 секунд
        print("Запуск main() на 45 секунд")
        task = asyncio.create_task(main())
        
        try:
            await asyncio.wait_for(task, timeout=main_task_duration)
        except asyncio.TimeoutError:
            task.cancel()
            print("main() завершена по таймеру")

        # Пауза перед следующим запускомвы
        if not stop_program:
            print(f"Пауза {pause_duration} секунд перед следующим запуском...")
            await asyncio.sleep(pause_duration)

        if not stop_program:
            print(f"Пауза {pause_duration} секунд перед следующим запуском...")

            # Запуск промежуточной задачи во время паузы
            intermediate_task = asyncio.create_task(perform_intermediate_task())

            # Ожидание перед следующим циклом
            await asyncio.sleep(pause_duration)

            # Остановка промежуточной задачи после паузы
            intermediate_task.cancel()
    
    cv2.destroyAllWindows()

# Главная асинхронная функция
async def main():
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
        "top": 168,    # Верхняя координата окна
        "left": 562,   # Левая координата окна
        "width": 767,  # Ширина окна (замените на ширину вашего экрана)
        "height": 570  # Высота окна (замените на высоту вашего экрана)607 было
    }

    # Запуск слушателя для нажатий клавиш

    while not stop_program:
        fixed_click_left_side = (random.randint(404, 533), random.randint(984, 985)) # first - x, second - y
        fixed_click_right_side = (random.randint(1377, 1493), random.randint(984, 985)) # first - x, second - y

        # random_click_left_side = (random.randint(80, 230), random.randint(236, 500))
        # random_click_right_side = (random.randint(1740, 1804), random.randint(236, 501))

        random_fixed_click_point = random.choice([
            fixed_click_left_side,
            fixed_click_right_side
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
                # vis_image = visualize_contours(capture_thresh, contours, matched_contour, bomb_contours)
                # save_screenshot(vis_image, prefix='matched')
            else:
                pass
                # Сохранение скриншота без совпадений
                # vis_image = visualize_contours(capture_thresh, contours, None, bomb_contours)
                # save_screenshot(vis_image, prefix='screen')
                # time.sleep(random.uniform(0.1, 0.11))
        else:
            click_fixed_point(*random_fixed_click_point)
            await asyncio.sleep(random.uniform(1, 1.5))
        cv2.destroyAllWindows()

async def main_loop():
    # Запускаем одновременно основной процесс и слушатель нажатий
    await asyncio.gather(
        run_main_for_duration(main_task_duration=45, pause_duration=15),
        stop_listener()
    )

if __name__ == '__main__':
    asyncio.run(main_loop())
