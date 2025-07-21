import argparse
from app.orchestrator import Orchestration

def main():
    parser = argparse.ArgumentParser(description="Customer short-listing agent")
    parser.add_argument("prompt", type=str, help="Natural-language requirement")
    args = parser.parse_args()

    orchestrator = Orchestration()
    result = orchestrator.run(args.prompt)
    print(result)

if __name__ == "__main__":
    main()
