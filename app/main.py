import argparse
import logging
from app.orchestrator import Orchestration
from app.config import settings

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orchestration.log'),
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser(description="Profile shortlisting agent with conditional routing")
    parser.add_argument("prompt", type=str, help="Natural-language requirement")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--max-retries", type=int, default=settings.MAX_RETRY_ATTEMPTS, help="Maximum retry attempts")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("Starting Profile Shortlisting with Conditional Routing...")
    print(f"Query: {args.prompt}")
    print(f"Max Retries: {args.max_retries}")
    print("-" * 60)
    
    orchestrator = Orchestration()
    result = orchestrator.run(args.prompt)
    
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    
    if isinstance(result, dict):
        print(f"Result: {result.get('result', 'No result')}")
        print(f"Processing Path: {' -> '.join(result.get('routing_history', []))}")
        print(f"Retry Count: {result.get('retry_count', 0)}")
        
        if result.get('error'):
            print(f"Error: {result['error']}")
        
        if result.get('quality_metrics'):
            print(f"Quality Metrics: {result['quality_metrics']}")
    else:
        print(result)

if __name__ == "__main__":
    main()
