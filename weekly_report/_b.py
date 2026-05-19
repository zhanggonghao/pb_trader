import os
BASE = os.getcwd()
def w(p, c):
    path = os.path.join(BASE, p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(c)
    print("OK:", p)
print("Builder loaded")