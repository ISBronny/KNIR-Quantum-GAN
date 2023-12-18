import os

import wandb
from dotenv import load_dotenv

load_dotenv()
wandb.login(relogin=True, key=os.getenv("WANDB_API_KEY", "606da00db4db699efabdef0dab836bbacb81e261"))

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'Generator Loss',
        'goal': 'minimize'
    },
    'parameters': {
        "learning_rate_discr": {
            'values': [0.002, 0.004]
        },
        "learning_rate_gen": {
            'values': [0.3]
        },
        'generator_opt': {
            'values': ['sgd']
        },
        'discriminator_opt': {
            'values': ['sgd']
        },
        "epochs": {
            'values': [8]
        },
        "image_size": {
            'values': [28]
        },
        "batch_size": {
            'values': [2]
        },
        "labels": {
            'values': [[0]]
        },
        "quantum_variables": {
            'values': [
                {
                    "n_qubits": 10,  # Total number of qubits / N
                    "q_depth": 1,  # Depth of the parameterised quantum circuit / D
                },
                {
                    "n_qubits": 10,  # Total number of qubits / N
                    "q_depth": 2,  # Depth of the parameterised quantum circuit / D
                },
                {
                    "n_qubits": 10,  # Total number of qubits / N
                    "q_depth": 3,  # Depth of the parameterised quantum circuit / D
                },
                {
                    "n_qubits": 10,  # Total number of qubits / N
                    "q_depth": 6,  # Depth of the parameterised quantum circuit / D
                },
                {
                    "n_qubits": 10,  # Total number of qubits / N
                    "q_depth": 8,  # Depth of the parameterised quantum circuit / D
                },
                {
                    "n_qubits": 10,  # Total number of qubits / N
                    "q_depth": 10,  # Depth of the parameterised quantum circuit / D
                },
                {
                    "n_qubits": 10,  # Total number of qubits / N
                    "q_depth": 12,  # Depth of the parameterised quantum circuit / D
                },
            ]
        }
    }
}

sweep_id = wandb.sweep(
    sweep=sweep_config,
    project='KNIR-Quantum-GAN'
)

print(sweep_id)