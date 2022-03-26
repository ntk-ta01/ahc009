ans = []
d = ["R", "L", "U", "D"]

for i in range(200):
    ans.append(d[i % 4])

print("".join(ans))
