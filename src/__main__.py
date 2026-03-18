"""CLI for vana."""
import sys, json, argparse
from .core import Vana

def main():
    parser = argparse.ArgumentParser(description="Vana — Deforestation Monitor. Satellite imagery analysis for real-time deforestation detection and alerts.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Vana()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.detect(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"vana v0.1.0 — Vana — Deforestation Monitor. Satellite imagery analysis for real-time deforestation detection and alerts.")

if __name__ == "__main__":
    main()
