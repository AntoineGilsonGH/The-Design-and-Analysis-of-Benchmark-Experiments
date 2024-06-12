import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of common Python packages
common_packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "jupyterlab",
    "notebook",
    "ipython",
    "sympy",
    "requests",
    "beautifulsoup4",
    "lxml",
    "pillow"
]

# Additional packages for machine learning and data science
ml_packages = [
    "tensorflow",
    "keras",
    "torch",
    "xgboost",
    "lightgbm",
    "catboost",
    "gensim",
    "nltk",
    "spacy",
    "sklearn"
]

# Install all packages
for package in common_packages + ml_packages:
    install(package)

print("All packages installed successfully.")

