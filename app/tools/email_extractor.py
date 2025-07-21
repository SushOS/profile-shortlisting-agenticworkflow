import re

EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

def extract_emails(text: str) -> list[str]:
    return list({m.group(0) for m in re.finditer(EMAIL_REGEX, text)})
