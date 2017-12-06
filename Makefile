# Makefile 
# Run `make` to clean taxons and content.
# Run `make init` to install required packages with pip.
# Run `make test` to run tests on pipeline_functions.
# Run `make clean` to remove all output files except raw data
# 	(i.e everything but the raw files downloaded from AWS)
# Run `make clean_all` to remove all output files and raw data

all : taxons content labelled
taxons : $(DATADIR)/clean_taxons.csv
content : $(DATADIR)/clean_content.csv
labelled : $(DATADIR)/labelled.csv

$(DATADIR)/new_content.csv : python/create_new.py $(DATADIR)/untagged_content.csv \
    $(DATADIR)/old_taxons.csv
	python3 python/create_new.py

$(DATADIR)/labelled.csv : python/create_labelled.py $(DATADIR)/raw_taxons.json \
    taxons content
	python3 python/create_labelled.py

$(DATADIR)/clean_taxons.csv : python/clean_taxons.py $(DATADIR)/raw_taxons.json
	python3 python/clean_taxons.py

$(DATADIR)/clean_content.csv : python/clean_content.py $(DATADIR)/raw_content.json.gz \
$(DATADIR)/document_type_group_lookup.json
	python3 python/clean_content.py

$(DATADIR)/document_type_group_lookup.json : 
	aws s3 cp $(S3BUCKET)/document_type_group_lookup.json $(DATADIR)/document_type_group_lookup.json

$(DATADIR)/raw_taxons.json : 
	aws s3 cp $(S3BUCKET)/raw_taxons.json $(DATADIR)/raw_taxons.json

$(DATADIR)/raw_content.json.gz :
	aws s3 cp $(S3BUCKET)/raw_content.json.gz $(DATADIR)/raw_content.json.gz

clean : 
	-rm -f $(DATADIR)/clean_taxons.csv $(DATADIR)/clean_content.csv \
	    $(DATADIR)/untagged_content.csv $(DATADIR)/empty_taxons.csv \
	    $(DATADIR)/labelled.csv $(DATADIR)/filtered.csv $(DATADIR)/old_taxons.csv \
	    $(DATADIR)/labelled_level1.csv $(DATADIR)/labelled_level2.csv \
	    $(DATADIR)/empty_taxons_not_world.csv $(DATADIR)/new_content.csv


clean_all : clean
	-rm -f $(DATADIR)/document_type_group_lookup.json \
	    $(DATADIR)/raw_taxons.json $(DATADIR)/raw_content.json.gz

init : 
	pip3 install -r python/requirements.txt

test : 
	cd python && python3 -m pytest

.PHONY : init test clean clean_all
