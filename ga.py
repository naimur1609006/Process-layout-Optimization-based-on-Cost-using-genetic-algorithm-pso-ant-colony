import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


file_path = 'D:/thesis/data/plant_layout_64x64_no_overlap.csv'
df_uploaded = pd.read_csv(file_path)


POP_SIZE = 50
N_UNITS = len(df_uploaded)
MAX_GENERATIONS = 200
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
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


def initialize_population():
    return [np.random.randint(0, GRID_SIZE, size=(N_UNITS, 2)) for _ in range(POP_SIZE)]

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
        NPi * PPi * PCDPi * FL
        for i in range(N_UNITS)
    )

    max_x, min_x = layout[:,0].max(), layout[:,0].min()
    max_y, min_y = layout[:,1].max(), layout[:,1].min()
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

def selection(pop):
    fitness = [calculate_fitness(ind) for ind in pop]
    selected = np.argsort(fitness)[:int(POP_SIZE * CROSSOVER_RATE)]
    return [pop[i] for i in selected]

def crossover(parent1, parent2):
    cp = random.randint(1, N_UNITS - 1)
    child1 = np.vstack((parent1[:cp], parent2[cp:]))
    child2 = np.vstack((parent2[:cp], parent1[cp:]))
    return child1, child2

def mutation(layout):
    if random.random() < MUTATION_RATE:
        i = random.randint(0, N_UNITS - 1)
        layout[i] = np.random.randint(0, GRID_SIZE, 2)
    return layout

# === Run GA ===
start_time = time.time()

population = initialize_population()
best_costs = []

final_cost = float('inf')
final_coords = None

for gen in range(MAX_GENERATIONS):
    selected = selection(population)
    next_gen = []

    for i in range(0, len(selected)-1, 2):
        p1, p2 = selected[i], selected[i+1]
        c1, c2 = crossover(p1, p2)
        next_gen.extend([mutation(c1), mutation(c2)])

    while len(next_gen) < POP_SIZE:
        next_gen.append(np.random.randint(0, GRID_SIZE, size=(N_UNITS, 2)))

    population = next_gen[:POP_SIZE]
    current_best_layout = min(population, key=calculate_fitness)
    current_best_cost = calculate_fitness(current_best_layout)

    if current_best_cost < final_cost:
        final_cost = current_best_cost
        final_coords = current_best_layout.copy()

    best_costs.append(final_cost)

end_time = time.time()
execution_time = end_time - start_time

# === Output ===
print(f"Final GA Total Cost: {final_cost:.2f} USD")
print(f"Final Coordinates:\n{final_coords}")
print(f"Execution Time: {execution_time:.2f} seconds")


plt.figure(figsize=(10, 5))
plt.plot(best_costs, marker='o', linestyle='-', color='darkblue')
plt.title('GA Cost Reduction Over Generations')
plt.xlabel('Generation')
plt.ylabel('Total Cost')
plt.grid(True)
plt.tight_layout()
plt.show()


unit_ids = df_uploaded['Layout Equipment Identification Code'].tolist()
plt.figure(figsize=(10, 8))

for idx, (x, y) in enumerate(final_coords):
    code = unit_ids[idx]
    if idx == 0:
        plt.scatter(x, y, c='green', s=150, marker='*', label='Start Unit')
    elif idx == len(final_coords) - 1:
        plt.scatter(x, y, c='blue', s=150, marker='X', label='End Unit')
    else:
        plt.scatter(x, y, c='red', s=100, marker='o')
    plt.text(x + 0.5, y + 0.5, f'{code}', fontsize=9)

    base_radius = df_uploaded.loc[idx, 'Damage radius (m)']
    scaled_radius = base_radius * 0.003
    risk_circle = patches.Circle((x, y), scaled_radius, facecolor='red', edgecolor='none', alpha=0.1)
    boundary_circle = patches.Circle((x, y), scaled_radius, facecolor='none', edgecolor='gray',
                                     linestyle='--', linewidth=1.2, alpha=0.8)
    plt.gca().add_patch(risk_circle)
    plt.gca().add_patch(boundary_circle)


for i in range(len(final_coords) - 1):
    x1, y1 = final_coords[i]
    x2, y2 = final_coords[i + 1]
    plt.annotate('', xy=(x2, y1), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle='->', color='gray', linestyle='--'))
    plt.annotate('', xy=(x2, y2), xytext=(x2, y1),
                 arrowprops=dict(arrowstyle='->', color='gray', linestyle='--'))

plt.title('Optimized Plant Layout (GA, 64×64) with Overlap Penalty')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(0, GRID_SIZE)
plt.ylim(0, GRID_SIZE)
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
