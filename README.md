# ML Pipeline Demo

A small demo project showing an ML workflow with preprocessing, training, and GitHub Actions automation.

## Structure

- `src/preprocessing.py` - data cleaning utilities
- `src/train.py` - training script that builds a simple model
- `tests/test_preprocessing.py` - unit tests for preprocessing
- `.github/workflows/ci.yml` - CI workflow that runs tests and training
- `requirements.txt` - project dependencies

## Run locally

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv/Scripts/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run tests:
   ```bash
   pytest tests -v
   ```
4. Train the model:
   ```bash
   python src/train.py
   ```
