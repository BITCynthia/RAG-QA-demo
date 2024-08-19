from pathlib import Path
from tclogger import OSEnver

SECRETS = OSEnver(Path(__file__).parent / "secrets.json")
OPENAI_ENVS = SECRETS["openai"]["gpt-4o"]
OPENAI_EMBEDDING_ENVS = SECRETS["openai"]["embedding"]
