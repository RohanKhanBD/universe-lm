from datasets import load_dataset
import os


def main() -> None:
    print("Downloading 1B pretraining data...")
    ds = load_dataset("vukrosic/blueberry-1B-pretrain")

    output_dir = "processed_data/pretrain_1B"
    os.makedirs(output_dir, exist_ok=True)
    ds.save_to_disk(output_dir)
    print(f"✅ Saved dataset to {output_dir}")


if __name__ == "__main__":
    main()
