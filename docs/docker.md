# Docker

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

All environment variables and volumes are described in [docker-compose.yaml](../docker-compose.yaml).
