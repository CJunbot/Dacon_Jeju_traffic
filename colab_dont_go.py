import pyautogui
import random
import time

while True:
    ans = random.randint(0,3)
    if ans == 0:
        seed = random.randint(-400,400)
        pyautogui.scroll(seed)
    else:
        seed = random.randint(-400, 400)
        pyautogui.scroll(seed)
        pyautogui.click(850,450)
    sleep = random.randint(60,120)
    time.sleep(sleep)
