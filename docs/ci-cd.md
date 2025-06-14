# CI/CD Pipeline

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

You can find the full pipeline config in [.github/workflows/ci.yml](../.github/workflows/ci.yml).
