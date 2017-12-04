# Makefile 
# Run `make` to clean taxons and content.
# Run `make init` to install required packages with pip.

all : taxons content labelled
taxons : $(DATADIR)/clean_taxons.csv
content : $(DATADIR)/clean_content.csv
labelled : $(DATADIR)/labelled.csv

$(DATADIR)/labelled.csv : python/labelled.py $(DATADIR)/raw_taxons.json taxons content
	python3 python/labelled.py

$(DATADIR)/clean_taxons.csv : python/clean_taxons.py $(DATADIR)/raw_taxons.json
	python3 python/clean_taxons.py

$(DATADIR)/clean_content.csv : python/clean_content.py $(DATADIR)/raw_content.json.gz 
	python3 python/clean_content.py

$(DATADIR)/raw_taxons.json : 
	aws s3 cp $(S3BUCKET)/raw_taxons.json $(DATADIR)/raw_taxons.json

$(DATADIR)/raw_content.json.gz :
	aws s3 cp $(S3BUCKET)/raw_content.json.gz $(DATADIR)/raw_content.json.gz

init : 
	pip3 install -r python/requirements.txt

.PHONY : init

