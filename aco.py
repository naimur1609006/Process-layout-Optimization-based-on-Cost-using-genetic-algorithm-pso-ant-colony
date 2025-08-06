import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches


file_path = 'D:/thesis/data/plant_layout_64x64_no_overlap.csv'  
df_uploaded = pd.read_csv(file_path)

N_UNITS = len(df_uploaded)
MAX_ITER = 200
NUM_ANTS = 50
ALPHA = 1
BETA = 2
RHO = 0.05
Q = 100
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


def calculate_fitness(layout):
    pic, poc = 0, 0
    for i in range(N_UNITS):
        for j in range(i+1, N_UNITS):
            xi, yi = layout[i]
            xj, yj = layout[j]
            Lij = abs(xj - xi) + abs(yj - yi)
            pic += Lij * CUP
            poc += Lij * CE * H

    eal = sum(
        df_uploaded.loc[i, 'Accident probability Pa(yr^-1)'] * df_uploaded.loc[i, 'Economic Value (EV) (E)'] +
        NPi * PPi * PCDPi * FL
        for i in range(N_UNITS)
    )

    max_x, min_x = layout[:, 0].max(), layout[:, 0].min()
    max_y, min_y = layout[:, 1].max(), layout[:, 1].min()
    land_area = (max_x - min_x) * (max_y - min_y)
    lc = t * ULC * land_area

    total_cost = pic + poc + eal + lc


    overlap_penalty = 0
    for i in range(N_UNITS):
        xi, yi = layout[i]
        Li = df_uploaded.loc[i, 'Length (m)']
        Wi = df_uploaded.loc[i, 'Width (m)']
        for j in range(i+1, N_UNITS):
            xj, yj = layout[j]
            Lj = df_uploaded.loc[j, 'Length (m)']
            Wj = df_uploaded.loc[j, 'Width (m)']
            if abs(xj - xi) < (Li + Lj) / 2 and abs(yj - yi) < (Wi + Wj) / 2:
                overlap_penalty += 1_000_000

    return total_cost + overlap_penalty


def run_aco():
    pheromone = np.ones((N_UNITS, 2))
    best_layout = None
    best_cost = np.inf
    best_scores = []

    for it in range(MAX_ITER):
        solutions = []
        costs = []

        for ant in range(NUM_ANTS):
            layout = np.zeros((N_UNITS, 2), dtype=int)

            for i in range(N_UNITS):
                x = np.random.randint(0, GRID_SIZE)  # ✅ For 64×64 grid
                y = np.random.randint(0, GRID_SIZE)
                layout[i] = [x, y]

            if random.random() < 0.5:
                i, j = np.random.randint(0, N_UNITS, 2)
                layout[[i, j]] = layout[[j, i]]

            cost = calculate_fitness(layout)
            solutions.append(layout)
            costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_layout = layout.copy()

        pheromone *= (1 - RHO)

        for layout, cost in zip(solutions, costs):
            for i in range(N_UNITS):
                pheromone[i, 0] += Q / cost
                pheromone[i, 1] += Q / cost

        best_scores.append(best_cost)

    return best_scores, best_layout


start_time = time.time()
best_scores, best_layout = run_aco()
end_time = time.time()

print("\n✅ ACO complete.")
print(f"Final Cost: {best_scores[-1]:.2f} USD")
print(f"Operation Time: {end_time - start_time:.2f} sec")
print(f"Best Layout:\n{best_layout}")


pd.DataFrame(best_layout, columns=['X', 'Y']).to_csv(
    'D:/thesis/data/aco_best_layout_64x64.csv', index=False)
print("✅ Final coordinates saved to CSV: aco_best_layout_64x64.csv")


plt.figure(figsize=(10, 5))
plt.plot(best_scores, marker='o', color='blue')
plt.title('ACO Cost Reduction over Iterations (64×64 Grid)')
plt.xlabel('Iteration')
plt.ylabel('Total Cost')
plt.grid(True)
plt.show()


unit_ids = df_uploaded['Layout Equipment Identification Code'].tolist()

plt.figure(figsize=(10, 8))

for idx, (x, y) in enumerate(best_layout):
    code = unit_ids[idx]

    if idx == 0:
        plt.scatter(x, y, c='green', s=150, marker='*', label='Start Unit')
    elif idx == len(best_layout) - 1:
        plt.scatter(x, y, c='blue', s=150, marker='X', label='End Unit')
    else:
        plt.scatter(x, y, c='red', s=100, marker='o')

    plt.text(x + 0.5, y + 0.5, f'{code}', fontsize=9)

    base_radius = df_uploaded.loc[idx, 'Damage radius (m)']
    scaled_radius = base_radius * 0.005  
    boundary_circle = patches.Circle(
        (x, y),
        scaled_radius,
        facecolor='red',
        alpha=0.1,
        edgecolor='gray',
        linestyle='--',
        linewidth=1.2
    )
    plt.gca().add_patch(boundary_circle)

for i in range(len(best_layout) - 1):
    x1, y1 = best_layout[i]
    x2, y2 = best_layout[i + 1]

    plt.annotate(
        '', xy=(x2, y1), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color='gray', linestyle='--')
    )
    plt.annotate(
        '', xy=(x2, y2), xytext=(x2, y1),
        arrowprops=dict(arrowstyle='->', color='gray', linestyle='--')
    )

plt.title('ACO Optimized Plant Layout with Risk Zones (64×64 Grid)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(0, GRID_SIZE)
plt.ylim(0, GRID_SIZE)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


