a = [-9, -8, 1, 3, -4, 6]
b = [-10, 4, 5, 6, -12]
tmp = max(a, b, key=lambda x: abs(x))
print(tmp)
