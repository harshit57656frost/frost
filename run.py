import os
import sys
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.5)
    print("\n[*] Opening web dashboard at http://127.0.0.1:5000 in your browser...")
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    from app import app
    threading.Thread(target=open_browser, daemon=True).start()
    print("=" * 60)
    print("SpamShield AI - Spam Mail Prediction System")
    print("Running on http://127.0.0.1:5000")
    print("Press Ctrl+C to terminate the server.")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
