"""Entry point for knir_quantum_gan."""
from knir_quantum_gan.gan import start_train
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    start_train()
