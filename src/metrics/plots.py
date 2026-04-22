import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use("ggplot")
def plot_cosine_matrix(cosine_matrix: np.ndarray, title: str) -> None:
    plt.figure()
    plt.imshow(cosine_matrix)
    plt.colorbar()
    plt.title(title, fontsize=14)
    plt.xlabel("Samples")
    plt.ylabel("Samples")
    plt.xticks([])
    plt.yticks([])
    os.makedirs("./results/plots/heatmaps", exist_ok=True)

    
    safe_title = title.replace(" ", "_").replace("/", "_").replace(":", "_")

    plt.savefig(f'./results/plots/heatmaps/{safe_title}.png', dpi=300, bbox_inches="tight")
    plt.show()

def plot_model_bars(results_by_scenario, dimensiones, model_name):
    
    scenarios = list(results_by_scenario.keys())
    x = np.arange(len(dimensiones))
    width = 0.8 / len(scenarios) if scenarios else 0.2 

    plt.figure(figsize=(10, 5))

    for i, scenario in enumerate(scenarios):
        impacts = [
            results_by_scenario[scenario][dim]["impact"] 
            for dim in dimensiones
        ]

        plt.bar(
            x + i * width,
            impacts,
            width=width,
            label=scenario,
            alpha=0.85
        )

    plt.xticks(x + (width * (len(scenarios) - 1) / 2), dimensiones, rotation=45)
    plt.ylabel("Impact (Intra - Inter)")
    plt.title(f"Dimension Impact per Scenario - {model_name}")
    plt.legend(title="Scenarios", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis="y", alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    save_path = "./analysis/plots/bar"
    os.makedirs(save_path, exist_ok=True)
    safe_title = model_name.replace(" ", "_").replace("/", "_")
    plt.savefig(f'{save_path}/{safe_title}.png', dpi=300, bbox_inches="tight")
    plt.show()