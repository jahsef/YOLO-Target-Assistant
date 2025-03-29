from screeninfo import get_monitors

monitor0 = get_monitors()[0]
print(monitor0.width)
print(monitor0.height)