# MathSolverExtension

MathSolverExtension uses the **Google Vision API** and **SymPy** to read and solve math equations directly from screenshots.

## Features

* Capture equations with the Snipping Tool (clipboard).
* Extract text using OCR (Google Vision API).
* Normalize math symbols and formatting.
* Solve equations, simplify expressions, or evaluate numeric results.

## Setup

1. Install dependencies:

```bash
pip install google-cloud-vision sympy pillow
```

2. Add your Google Cloud Vision API key path to `KEY_PATH` in the script.

3. Run the program:

```bash
python mathsolver.py
```

## Usage

1. Start the program.
2. Take a screenshot with **Win + Shift + S**.
3. Press **Enter** in the terminal to see the solution.