"""Entry point for knir_quantum_gan."""
import os

import wandb

from gan import start_train
from dotenv import load_dotenv


if __name__ == "__main__":
    wandb.login(relogin=True, key=os.getenv("WANDB_API_KEY", "606da00db4db699efabdef0dab836bbacb81e261"))
    load_dotenv()
    wandb.agent('jch3zjdi', function=start_train, count=30, project='KNIR-Quantum-GAN')