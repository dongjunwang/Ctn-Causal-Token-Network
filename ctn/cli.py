import argparse
import json
from .simulator.generator import generate_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate CTN first-stage datasets")
    parser.add_argument("--output", default="data/generated")
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(generate_dataset(args.output, args.train, args.validation, args.test, args.seed), ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
