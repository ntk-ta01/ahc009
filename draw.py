# cargo run --release --bin ahc009-a < ./tools/in/0000.txt ^ draw0000.txt
# python3 draw.py
import matplotlib.pyplot as plt

temp = []
val = []
with open("./draw0000.txt") as f:
    for line in f:
        li = [float(i) for i in line.split()]
        if len(li) < 2:
            continue
        a, b = li
        temp.append(a)
        val.append(b)

# plt.rcParams['font.family'] = "MS Gothic"

fig = plt.figure()
ax = fig.add_subplot(111)

n = len(temp)
ax.plot(temp[n//2:], val[n//2:])
ax.invert_xaxis()
ax.set_xlabel("temp", size=14, weight="light")
ax.set_ylabel("objective function value", size=14, weight="light")
ax.grid(which="major", axis="x", color="black", alpha=0.25,
        linestyle="--", linewidth=1)
ax.grid(which="major", axis="y", color="black", alpha=0.25,
        linestyle="--", linewidth=1)
plt.savefig("graph.png")
