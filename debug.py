import starry
import numpy as np
import matplotlib.pyplot as plt

ydeg = 15

print("1")
map = starry.Map(ydeg, lazy=True)
print("2")
map.load("earth")
print("3")
x = map.render(projection="moll").eval()
print(x)
print("4")


print("1")
map = starry.Map(ydeg, lazy=False)
print("2")
map.load("earth")
print("3")
x = map.render(projection="moll")
print(x)
print("4")
