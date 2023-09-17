import os

import wandb
from dotenv import load_dotenv

load_dotenv()
wandb.login(relogin=True, key=os.getenv("WANDB_API_KEY", "606da00db4db699efabdef0dab836bbacb81e261"))

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'Generator Loss',
        'goal': 'minimize'
    },
    'parameters': {
        "learning_rate_discr": {
            'values': [0.02]
        },
        "learning_rate_gen": {
            'values': [0.6]
        },
        'generator_opt': {
            'values': ['sgd']
        },
        'discriminator_opt': {
            'values': ['sgd']
        },
        "epochs": {
            'values': [6]
        },
        "image_size": {
            'values': [28]
        },
        "batch_size": {
            'values': [1]
        },
        "labels": {
            'values': [[0]]
        },
        "quantum_variables": {
            'values': [
                {
                    "n_qubits": 6,  # Total number of qubits / N
                    "n_a_qubits": 2,  # Number of ancillary qubits / N_A
                    "q_depth": 8,  # Depth of the parameterised quantum circuit / D
                    "n_generators": 49
                },
                {
                    "n_qubits": 5,  # Total number of qubits / N
                    "n_a_qubits": 1,  # Number of ancillary qubits / N_A
                    "q_depth": 6,  # Depth of the parameterised quantum circuit / D
                    "n_generators": 49
                },
                {
                    "n_qubits": 5,  # Total number of qubits / N
                    "n_a_qubits": 1,  # Number of ancillary qubits / N_A
                    "q_depth": 8,  # Depth of the parameterised quantum circuit / D
                    "n_generators": 49
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