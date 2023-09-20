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
            'values': [0.0025]
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
            'values': [10]
        },
        "image_size": {
            'values': [8]
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
                    "n_a_qubits": 1,  # Number of ancillary qubits / N_A
                    "q_depth": 1,  # Depth of the parameterised quantum circuit / D
                    "n_generators": 14
                },
                {
                    "n_qubits": 6,  # Total number of qubits / N
                    "n_a_qubits": 1,  # Number of ancillary qubits / N_A
                    "q_depth": 2,  # Depth of the parameterised quantum circuit / D
                    "n_generators": 14
                },
                # {
                #     "n_qubits": 5,
                #     "n_a_qubits": 1,
                #     "q_depth": 2,
                #     "n_generators": 4
                # },
                # {
                #     "n_qubits": 5,
                #     "n_a_qubits": 1,
                #     "q_depth": 3,
                #     "n_generators": 4
                # },
                # {
                #     "n_qubits": 5,
                #     "n_a_qubits": 1,
                #     "q_depth": 4,
                #     "n_generators": 4
                # }
            ]
        }
    }
}

sweep_id = wandb.sweep(
    sweep=sweep_config,
    project='KNIR-Quantum-GAN'
)

print(sweep_id)