import os
import uvicorn

port = int(os.environ.get("PORT", 8000))
print(f"Avvio FootballHub su porta {port}")
uvicorn.run("main:app", host="0.0.0.0", port=port)
