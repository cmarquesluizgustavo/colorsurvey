#!/usr/bin/env python3
"""Generate all configuration files for 3rd_experiments."""
import json
import os

# Base configuration template
def create_config(
    experiment_name,
    top_n_colors,
    balance_strategy,
    fixed_samples_per_class,
    loss_fn,
    embedding_dim,
    margin=None,
    temperature=None
):
    """Create a configuration dictionary for a metric learning experiment."""
    config = {
        "trainer_type": "metric_learning",
        "experiment_name": experiment_name,
        "seed": 42,
        "data": {
            "csv_path": "mainsurvey_data.csv",
            "top_n_colors": top_n_colors,
            "test_size": 0.2,
            "balance_strategy": balance_strategy,
            "batch_size": 256
        },
        "model": {
            "choice_model_type": "mlp",
            "embedding_dim": embedding_dim,
            "hidden_dim": embedding_dim * 2
        },
        "training": {
            "loss_fn": loss_fn,
            "cycles": 5,
            "epochs_per_step": 2,
            "lr_embedding": 0.001,
            "lr_classifier": 0.001,
            "lambda": 0.1,
            "batch_size": 256,
            "early_stopping": {
                "metric": "youdens_j",
                "patience": 3,
                "min_delta": 0.001,
                "at_cycle_end": True
            }
        }
    }
    
    # Add fixed_samples_per_class if using balanced_fixed
    if balance_strategy == "balanced_fixed":
        config["data"]["fixed_samples_per_class"] = fixed_samples_per_class
    
    # Add loss-specific parameters
    if loss_fn == "conditional_triplet":
        config["training"]["margin"] = margin
    elif loss_fn == "snnl":
        config["training"]["temperature"] = temperature
    
    return config


# Define experiment parameters
data_configs = [
    {"colors": 15, "strategy": "none", "fixed_samples": None},
    {"colors": 15, "strategy": "balanced_fixed", "fixed_samples": 82039},
    {"colors": 129, "strategy": "none", "fixed_samples": None},
    {"colors": 129, "strategy": "balanced_fixed", "fixed_samples": 15531},
]

loss_configs = [
    {"loss": "conditional_triplet", "margins": [0.5, 1.0]},
    {"loss": "snnl", "temperatures": [0.1, 1.0]},
]

embedding_dims = [4, 8, 16, 32]

# Generate all configurations
configs = []
for data_cfg in data_configs:
    for loss_cfg in loss_configs:
        for emb_dim in embedding_dims:
            if loss_cfg["loss"] == "conditional_triplet":
                for margin in loss_cfg["margins"]:
                    exp_name = (
                        f"ml_mlp_{data_cfg['colors']}colors_{'unbalanced' if data_cfg['strategy'] == 'none' else data_cfg['strategy']}_"
                        f"ctriplet_m{margin}_dim{emb_dim}"
                    )
                    config = create_config(
                        experiment_name=exp_name,
                        top_n_colors=data_cfg["colors"],
                        balance_strategy=data_cfg["strategy"],
                        fixed_samples_per_class=data_cfg["fixed_samples"],
                        loss_fn=loss_cfg["loss"],
                        embedding_dim=emb_dim,
                        margin=margin,
                    )
                    configs.append((exp_name, config))
            else:  # snnl
                for temp in loss_cfg["temperatures"]:
                    exp_name = (
                        f"ml_mlp_{data_cfg['colors']}colors_{'unbalanced' if data_cfg['strategy'] == 'none' else data_cfg['strategy']}_"
                        f"snnl_t{temp}_dim{emb_dim}"
                    )
                    config = create_config(
                        experiment_name=exp_name,
                        top_n_colors=data_cfg["colors"],
                        balance_strategy=data_cfg["strategy"],
                        fixed_samples_per_class=data_cfg["fixed_samples"],
                        loss_fn=loss_cfg["loss"],
                        embedding_dim=emb_dim,
                        temperature=temp,
                    )
                    configs.append((exp_name, config))

# Save all configurations
script_dir = os.path.dirname(os.path.abspath(__file__))
for exp_name, config in configs:
    filename = os.path.join(script_dir, f"{exp_name}.json")
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Created: {exp_name}.json")

print(f"\n✅ Generated {len(configs)} configuration files")

# Generate experiments.txt entries
print("\n📝 Entries for experiments.txt:")
for exp_name, _ in configs:
    print(f"3rd_experiments/{exp_name}.json")
