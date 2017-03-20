.PHONY: web, install

web:
	python server.py

install:
	pip install -r requirements.txt