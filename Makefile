# Makefile 
# Run `make` to clean taxons and content.
# Run `make init` to install required packages with pip.

all : taxons content
taxons : data/clean_taxons.csv
content : data/clean_content.csv

data/clean_taxons.csv : python/clean_taxons.py data/raw_taxons.json
	python3 python/clean_taxons.py

data/clean_content.csv : python/clean_content.py data/raw_content.json.gz 
	python3 python/clean_content.py

data/raw_taxons.json : 
	aws s3 cp s3://buod-govuk-taxonomy-supervised-learning/raw_taxons.json ./data/raw_taxons.json

data/raw_content.json.gz : 
	aws s3 cp s3://buod-govuk-taxonomy-supervised-learning/raw_content.json.gz ./data/raw_content.json.gz

.PHONY :

init :
	pip3 install -r python/requirements.txt


