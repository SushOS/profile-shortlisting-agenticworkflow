import pandas as pd
from datetime import datetime
import os
import csv

# Set pandas options to ensure full text display
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

class SheetWriter:
    def __init__(self):
        # Always use CSV output - no Google Sheets dependency
        print("SheetWriter initialized for CSV output")

    def write_dataframe(self, df: pd.DataFrame, tab: str = "Results"):
        # Create filename with timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"profile_results_{tab.lower()}_{timestamp}.csv"
        
        # Save to CSV file with proper quoting to preserve full text
        df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"Results saved to {filename}")
        print(f"Found {len(df)} profiles")
        
        # Also print a sample of the data to verify snippet content
        if not df.empty and 'snippet' in df.columns:
            print(f"Sample snippet length: {len(df.iloc[0]['snippet'])} characters")
            
        return filename
