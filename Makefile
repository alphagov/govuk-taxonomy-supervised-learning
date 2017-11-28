# Makefile 
# Run `make` to clean taxons and content.
# Run `make init` to install required packages with pip.

all : taxons content
taxons : data/clean_taxons.csv
content : data/clean_content.csv

data/clean_taxons.csv : python/clean_taxons.py
	python python/clean_taxons.py

data/clean_content.csv : python/clean_content.py
	python python/clean_content.py

.PHONY :

init :
	pip install -r python/requirements.txt


