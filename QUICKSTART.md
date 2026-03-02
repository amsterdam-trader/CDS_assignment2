# Quick Start Guide

## First Time Setup

### Windows Users
1. Double-click `setup_env.bat`
2. Wait for installation to complete
3. Open `main.ipynb` in Jupyter Notebook/Lab

### Mac/Linux Users
1. Open terminal in this directory
2. Run: `chmod +x setup_env.sh && ./setup_env.sh`
3. Open `main.ipynb` in Jupyter Notebook/Lab

## Running the Analyses

### Option 1: Using Jupyter Notebook (Recommended)
1. Activate the environment:
   - Windows: `venv\Scripts\activate.bat`
   - Mac/Linux: `source venv/bin/activate`
2. Start Jupyter: `jupyter notebook main.ipynb`
3. Run all cells (Cell → Run All)

### Option 2: Running Individual Questions from Command Line
```bash
# Activate environment first (see above)

# Run Question 1 (Nonparametric trend estimation)
python -m src.q1

# Run Question 2 (Time-varying regression)
python -m src.q2

# Run Question 3 (t-location model)
python src/q3.py
```

## Path Configuration

The project uses relative paths, so everything works as long as you:
- Keep the project structure intact
- Run scripts/notebooks from the project root directory
- Use the virtual environment created by `setup_env`

**In main.ipynb:** The first code cell automatically adds `src/` to Python's path, so imports work correctly.

**In scripts:** Each script uses `Path(__file__).resolve().parents[1]` to find the project root, then constructs paths relative to it.

## Output Files

All figures are automatically saved to:
- `figures_q1/` - Question 1 outputs
- `figures_q2/` - Question 2 outputs  

## Troubleshooting

**"Module not found" errors:**
- Make sure you activated the virtual environment
- In Jupyter, restart the kernel and run the first cell again

**"File not found" errors:**
- Ensure you're running from the project root directory
- Check that `data/` folder exists with the required files

**Import errors:**
- Verify requirements are installed: `pip list`
- Reinstall if needed: `pip install -r requirements.txt`

## Package Versions

The code has been tested with:
- Python 3.8+
- NumPy ≥1.21.0
- SciPy ≥1.7.0
- Matplotlib ≥3.4.0
- Pandas ≥1.3.0
- Statsmodels ≥0.13.0
