.PHONY: help build up down restart logs shell test

help:
	@echo "Available commands:"
	@echo "  build        Build Docker images"
	@echo "  up           Start all services"
	@echo "  down         Stop all services"
	@echo "  restart      Restart services"
	@echo "  logs         View logs"
	@echo "  shell        Open a shell in the API container"
	@echo "  test         Run tests"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

restart: down up

logs:
	docker-compose logs -f

shell:
	docker-compose exec api bash

test:
	docker-compose exec api pytest tests/