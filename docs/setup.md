# Setup

## Quick Start

1. Clone the repository and go to the project folder:

   ```powershell
   git clone <repo_url>
   cd <project_folder>
   ```

2. Install Poetry (if not installed):

   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   ```

3. Install all dependencies:

   ```powershell
   poetry install
   ```

4. Install pre-commit hooks (once):

   ```powershell
   poetry run pre-commit install
   ```

5. Main development commands (run each separately using Poetry):

   * Code formatting:

     ```powershell
     poetry run black .
     poetry run ruff format
     ```
   * Linting and type checking:

     ```powershell
     poetry run black . --check
     poetry run mypy mlops/
     ```
   * Run tests:

     ```powershell
     poetry run pytest --cov=mlops
     ```
   * Run pre-commit hooks manually:

     ```powershell
     poetry run pre-commit run --all-files
     ```
   * Quick app run:

     ```powershell
     poetry run python -m mlops
     ```
   * Activate Poetry environment:

     ```powershell
     poetry shell
     ```

## Dependencies

Poetry is used for dependency management:

```powershell
poetry install                 # Install all dependencies
poetry add <lib>               # Add a library
poetry add --group dev <lib>   # Add a dev dependency
```

---

## Data Version Control (DVC)

This project uses [Data Version Control (DVC)](https://dvc.org/) to manage and version datasets and machine learning models.

### Getting Started with DVC

1. Install DVC (if not already installed):

   ```bash
   poetry add --group dev dvc
   ```
2. Initialize DVC in the project (do once):

   ```bash
   dvc init
   git add .dvc .dvcignore
   git commit -m "Initialize DVC"
   ```

### Versioning Data and Models

1. Add raw and processed data to DVC:

   ```bash
   dvc add data/raw/train.csv data/raw/test.csv
   ```

   This creates `.dvc` tracking files for your datasets.
2. Commit the DVC metafiles and gitignore updates:

   ```bash
   git add data/raw/*.dvc data/processed/*.dvc models/*.dvc .gitignore
   git commit -m "Track data and models with DVC"
   ```
3. (Optional) Push data to remote storage (e.g., S3, Google Drive):

   ```bash
   dvc push
   ```
4. To pull the latest datasets and models:

   ```bash
   dvc pull
   ```

**Always commit the generated `.dvc` files to Git** to track the versions of your datasets and models, while keeping the actual large files outside the Git repository.
This enables **team collaboration** and **experiment reproducibility**.
