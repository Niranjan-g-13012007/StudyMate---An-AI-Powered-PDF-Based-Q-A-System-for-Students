try:
    import flask
    print("Flask is installed and importable")
    print(f"Flask version: {flask.__version__}")
except ImportError as e:
    print(f"Error importing Flask: {e}")

try:
    import torch
    print("PyTorch is installed and importable")
    print(f"PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"Error importing PyTorch: {e}")

print("\nPython path:")
import sys
for path in sys.path:
    print(f"- {path}")
