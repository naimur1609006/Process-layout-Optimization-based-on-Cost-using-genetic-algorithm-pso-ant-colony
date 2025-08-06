import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time  


file_path = "D:/thesis/data/plant_layout_64x64_no_overlap.csv"
df_uploaded = pd.read_csv(file_path)

N_UNITS = len(df_uploaded)
SWARM_SIZE = 50
MAX_ITER = 100
GRID_SIZE = 64

CUP = 100
CE = 0.1
H = 6000
FL = 2e6
s = 0.05
T = 10
t = (s * (1 + s)**T) / ((1 + s)**T - 1)
ULC = 15
NPi = 1
PPi = 0.5
PCDPi = 0.05


w = 0.5
c1 = 1.5
c2 = 1.5


def calculate_fitness(layout):
    pic = poc = 0
    for i in range(N_UNITS):
        for j in range(i+1, N_UNITS):
            xi, yi = layout[i]
            xj, yj = layout[j]
            Lij = abs(xj - xi) + abs(yj - yi)
            pic += Lij * CUP
            poc += Lij * CE * H

    eal = sum(
        df_uploaded.loc[i, 'Accident probability Pa(yr^-1)'] * df_uploaded.loc[i, 'Economic Value (EV) (E)'] +
        NPi * PPi * PCDPi * FL for i in range(N_UNITS)
    )

    max_x, min_x = layout[:, 0].max(), layout[:, 0].min()
    max_y, min_y = layout[:, 1].max(), layout[:, 1].min()
    land_area = (max_x - min_x) * (max_y - min_y)
    lc = t * ULC * land_area

    total_cost = pic + poc + eal + lc


    overlap_penalty = 0
    for i in range(N_UNITS):
        xi, yi = layout[i]
        Li, Wi = df_uploaded.loc[i, 'Length (m)'], df_uploaded.loc[i, 'Width (m)']
        for j in range(i+1, N_UNITS):
            xj, yj = layout[j]
            Lj, Wj = df_uploaded.loc[j, 'Length (m)'], df_uploaded.loc[j, 'Width (m)']
            if abs(xj - xi) < (Li + Lj) / 2 and abs(yj - yi) < (Wi + Wj) / 2:
                overlap_penalty += 1_000_000

    return total_cost + overlap_penalty


X = [np.random.randint(0, GRID_SIZE, size=(N_UNITS, 2)) for _ in range(SWARM_SIZE)]
V = [np.random.uniform(-1, 1, size=(N_UNITS, 2)) for _ in range(SWARM_SIZE)]

pBest = X.copy()
pBest_scores = [calculate_fitness(x) for x in pBest]

gBest_idx = np.argmin(pBest_scores)
gBest = pBest[gBest_idx].copy()
gBest_score = pBest_scores[gBest_idx]
best_scores = []


start_time = time.time()


for iter in range(MAX_ITER):
    for i in range(SWARM_SIZE):
        r1 = np.random.rand(N_UNITS, 2)
        r2 = np.random.rand(N_UNITS, 2)

        V[i] = (w * V[i] +
                c1 * r1 * (pBest[i] - X[i]) +
                c2 * r2 * (gBest - X[i]))

        X[i] = X[i] + V[i]
        X[i] = np.clip(X[i], 0, GRID_SIZE - 1).astype(int)

        fit = calculate_fitness(X[i])
        if fit < pBest_scores[i]:
            pBest[i] = X[i].copy()
            pBest_scores[i] = fit

    best_idx = np.argmin(pBest_scores)
    if pBest_scores[best_idx] < gBest_score:
        gBest = pBest[best_idx].copy()
        gBest_score = pBest_scores[best_idx]

    best_scores.append(gBest_score)

end_time = time.time()
execution_time = end_time - start_time


print(f"✅ Final PSO Total Cost (64×64): {gBest_score:.2f} USD")
print(f"✅ Final PSO Coordinates:\n{gBest}")
print(f"⏱️ Execution Time: {execution_time:.2f} seconds")


plt.figure(figsize=(10, 5))
plt.plot(best_scores, marker='o', color='purple')
plt.title('PSO Cost Reduction over Iterations (With Overlap Penalty, 64×64)')
plt.xlabel('Iteration')
plt.ylabel('Total Cost')
plt.grid(True)
plt.show()

unit_ids = df_uploaded['Layout Equipment Identification Code'].tolist()
plt.figure(figsize=(10, 8))

for idx, (x, y) in enumerate(gBest):
    code = unit_ids[idx]
    color = 'green' if idx == 0 else 'blue' if idx == len(gBest) - 1 else 'red'
    marker = '*' if idx == 0 else 'X' if idx == len(gBest) - 1 else 'o'
    size = 150 if idx in [0, len(gBest) - 1] else 100
    plt.scatter(x, y, c=color, s=size, marker=marker)
    plt.text(x + 0.5, y + 0.5, f'{code}', fontsize=9)

    base_radius = df_uploaded.loc[idx, 'Damage radius (m)']
    scaled_radius = base_radius * 0.003
    risk_circle = patches.Circle((x, y), scaled_radius, facecolor='red', edgecolor='none', alpha=0.1)
    plt.gca().add_patch(risk_circle)
    boundary_circle = patches.Circle((x, y), scaled_radius, facecolor='none', edgecolor='gray',
                                     linestyle='--', linewidth=1.2, alpha=0.8)
    plt.gca().add_patch(boundary_circle)


for i in range(len(gBest) - 1):
    x1, y1 = gBest[i]
    x2, y2 = gBest[i + 1]
    plt.annotate('', xy=(x2, y1), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='gray', linestyle='--'))
    plt.annotate('', xy=(x2, y2), xytext=(x2, y1), arrowprops=dict(arrowstyle='->', color='gray', linestyle='--'))

plt.title('Optimized Plant Layout (PSO, 64×64) with Overlap Check')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(0, GRID_SIZE)
plt.ylim(0, GRID_SIZE)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()








