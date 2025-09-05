.PHONY: initial program setup, start_dev start_prod docker_up docker_prod logs

start_dev:
	@echo "Running app in development.."
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8089

start_prod:
	@echo "Running app in production..."
	poetry run gunicorn -w 2 -k uvicorn.workers.UvicornWorker src.main:app -b 0.0.0.0:8088

run_postgresdb:
	docker-compose -f docker-compose.yml up


docker_up:
	@echo "Starting Docker Compose (dev/prod combined)..."
	docker-compose up

docker_down:
	@echo "Shutting down dev database..."
	docker-compose down

docker_prod:
	@echo "Starting only the production container..."
	docker-compose up -d --build

logs:
	docker-compose logs -f