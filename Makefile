.PHONY: web, install, data

web:
	python server.py

install:
	pip install -r requirements.txt

data:
	./get_data.sh