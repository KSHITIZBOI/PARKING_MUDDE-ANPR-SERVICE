"""
Entry point for ANPR Service
Run this file to start the server
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("=" * 60)
    print(f"🚀 Starting {settings.SERVICE_NAME}")
    print(f"📍 Version: {settings.VERSION}")
    print(f"🌐 Server: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"📖 Docs: http://localhost:{settings.API_PORT}/docs")
    print("=" * 60)

    uvicorn.run(
        "app.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )
