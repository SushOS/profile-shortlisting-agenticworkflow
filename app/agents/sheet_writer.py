class SheetWriterAgent:
    def __init__(self):
        from app.tools.google_sheets import SheetWriter
        self.writer = SheetWriter()

    def run(self, df):
        self.writer.write_dataframe(df)
        return f"Wrote {len(df)} rows to Google Sheet ✅"
