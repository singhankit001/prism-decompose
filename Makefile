.PHONY: help install install-full run test lint docker docker-up clean

help:
	@echo "make install       core dependencies (weights-free path)"
	@echo "make install-full  plus neural backends (rembg, easyocr, LaMa)"
	@echo "make run           start the server on :7860"
	@echo "make test          run the test suite"
	@echo "make docker-up     build and run via docker compose"
	@echo "make clean         remove caches and build artefacts"

install:
	python -m pip install -r requirements.txt

install-full: install
	python -m pip install easyocr simple-lama-inpainting

run:
	python app.py

test:
	python -m pytest tests/ -q

test-classical:
	LAYERFORGE_CLASSICAL=1 python -m pytest tests/ -q

docker:
	docker build -t prism .

docker-up:
	docker compose up --build

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info build dist
