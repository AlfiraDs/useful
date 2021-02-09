import pyautogui
from tqdm import trange
import time

pyautogui.FAILSAFE = False
i = 0
secs = 100
while True:
    pyautogui.click(500, 500)
    i += 1
    desc = 'Sleeping for %s sec. Already clicked %d times' % (secs, i)
    t = trange(secs, desc=desc, leave=False)
    for sec in t:
        time.sleep(1)
    t.set_description(desc=desc)
    t.refresh()
