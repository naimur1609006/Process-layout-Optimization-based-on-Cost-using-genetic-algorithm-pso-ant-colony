import pandas as pd
import matplotlib.pyplot as plt

results_data = {
    "Algorithm": ["Genetic Algorithm", "Particle Swarm Optimization", "Ant Colony Optimization"],
    "Final Cost (USD)": [5128742.19, 3463863.81, 5974002.83],
    "Execution Time (s)": [58.65, 49.82, 41.56]
}


df = pd.DataFrame(results_data)


best_cost_idx = df["Final Cost (USD)"].idxmin()
best_time_idx = df["Execution Time (s)"].idxmin()


print("📊 Algorithm Comparison Table:\n")
print(df.to_string(index=False))
print(f"\n✅ Lowest Cost: {df.loc[best_cost_idx, 'Algorithm']}")
print(f"✅ Fastest Execution: {df.loc[best_time_idx, 'Algorithm']}")


fig, axs = plt.subplots(1, 2, figsize=(14, 6))
plt.suptitle("Optimization Algorithm Performance Comparison", fontsize=16, fontweight='bold')


colors_cost = ['gray'] * len(df)
colors_cost[best_cost_idx] = 'green'

axs[0].bar(df["Algorithm"], df["Final Cost (USD)"], color=colors_cost)
axs[0].set_title("Final Cost", fontsize=14)
axs[0].set_ylabel("Total Cost ")
axs[0].grid(True, axis='y')

for i, cost in enumerate(df["Final Cost (USD)"]):
    axs[0].text(i, cost + 50000, f"${cost/1e6:.2f}M", ha='center', fontsize=10)


colors_time = ['gray'] * len(df)
colors_time[best_time_idx] = 'blue'

axs[1].bar(df["Algorithm"], df["Execution Time (s)"], color=colors_time)
axs[1].set_title("Execution Time", fontsize=14)
axs[1].set_ylabel("Time (seconds)")
axs[1].grid(True, axis='y')

for i, time in enumerate(df["Execution Time (s)"]):
    axs[1].text(i, time + 1.5, f"{time:.1f}s", ha='center', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
