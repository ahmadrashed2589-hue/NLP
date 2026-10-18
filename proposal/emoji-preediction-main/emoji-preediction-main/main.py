from src.utils import create_folders, set_seed
from src.baseline_model import train_baseline


def main():
    create_folders()
    set_seed(42)

    print("Starting Emoji Prediction Project...")
    print("Training baseline model...")

    train_baseline(
        train_path="data/Train.csv",
        test_path="data/Test.csv",
        mapping_path="data/Mapping.csv",
        top_n=20
    )

    print("Project completed successfully.")


if __name__ == "__main__":
    main()