# MLOps Project

---

## GitHub Flow

This project follows the standard GitHub Flow for collaborative development:

1. **Create a branch:**

   ```bash
   git checkout main
   git pull
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and commit:**

   ```bash
   git add .
   git commit -m "Short description of the changes"
   ```

3. **Open a Pull Request:**

   ```bash
   git push -u origin feature/your-feature-name
   ```

   * Go to GitHub → Pull Requests → New Pull Request
   * Select your branch, create a PR

4. **Wait for review and CI:**
   Wait for automatic checks to complete and address any comments.

5. **Merge:**
   After approval, merge to main (usually via "Squash and merge").

6. **Delete the branch:**

   ```bash
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```

---

## Code Quality and Linting

The project uses modern tools for code quality:

* **Black** — automatic code formatter
* **Ruff** — fast linter (replaces Flake8, isort, etc.)
* **MyPy** — static type checking
* **Pre-commit** — automatic checks before each commit
* **pytest/pytest-cov** — tests and coverage

---

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

   * **Code formatting:**

     ```powershell
     poetry run black .
     poetry run ruff format
     ```

   * **Linting and type checking:**

     ```powershell
     poetry run black . --check
     poetry run mypy mlops/
     ```

   * **Run tests:**

     ```powershell
     poetry run pytest --cov=mlops
     ```

   * **Run pre-commit hooks manually:**

     ```powershell
     poetry run pre-commit run --all-files
     ```

   * **Quick app run:**

     ```powershell
     poetry run python -m mlops
     ```

   * **Activate Poetry environment:**

     ```powershell
     poetry shell
     ```

---

## Dependencies

Poetry is used for dependency management:

```powershell
poetry install                 # Install all dependencies
poetry add <lib>               # Add a library
poetry add --group dev <lib>   # Add a dev dependency
```

---

## Running with Docker

You can also use Docker to run the project.

### Using Docker Compose (recommended):

**For development (interactive mode):**

```bash
docker compose build dev
docker compose run --rm dev
```

**For production:**

```bash
docker compose build app
docker compose up app
```

### Using plain Docker:

**Development:**

```bash
docker build --target development -t mlops-dev .
docker run -it --rm mlops-dev
```

**Production:**

```bash
docker build --target production -t mlops .
docker run -it --rm mlops
```

All environment variables and volumes are described in [docker-compose.yaml](./docker-compose.yaml).

---

## CI/CD Pipeline

A full CI/CD pipeline is configured using **GitHub Actions**:

* **Linting and static checks:** Black, Ruff, and MyPy are run on every PR, push, and release tag.
* **Testing:** All unit tests are run with pytest and coverage is collected.
* **Python package build:** The package is built using Poetry.
* **Docker image build and publish:** The Docker image is pushed to [GitHub Container Registry (ghcr.io)](https://ghcr.io).
* **Documentation publishing:** Documentation is automatically deployed to GitHub Pages.

**The pipeline is triggered on:**

* Pull Requests to `main`
* Pushes to `main`
* Tag pushes like `v*` (releases)

---
