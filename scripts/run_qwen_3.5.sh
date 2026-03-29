import sys
import os

# Get the path to model.py relative to this script
model_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models", "Qwen-3.5-35B-A3B", "model.py"
)

# Run model.py as a script
if __name__ == "__main__":
    with open(model_path, "r") as f:
        code = f.read()
    exec(compile(code, model_path, 'exec'))