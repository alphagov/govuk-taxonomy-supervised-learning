# Makefile
.DEFAULT_GOAL := all
all: data/clean_taxons.csv data/clean_content.csv

data/clean_taxons.csv: python/clean_taxons.py
	python python/clean_taxons.py

data/clean_content.csv: python/clean_content.py
	python python/clean_content.py

