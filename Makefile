# Makefile
# Run `make` to clean taxons and content.
# Run `make pip_install` to install required packages with pip.
# Run `make check` to run tests on pipeline_functions.
# Run `make clean` to remove all output files except raw data
# 	(i.e everything but the raw files downloaded from AWS)
# Run `make clean_all` to remove all output files and raw data
# Run `make upload` to upload all final training sets to S3 bucket
# (note that you will need the necessary write permissions to do this)

# Makefile cheatsheet:
#
#   $< means the first prerequsite
#   $@ means the target

all : taxons content labelled
taxons : $(DATADIR)/clean_taxons.csv.gz
content : $(DATADIR)/clean_content.csv.gz
labelled : $(DATADIR)/labelled.csv.gz
new: $(DATADIR)/new_content.csv.gz
export_all: data/export_filtered_content.json.gz data/export_untagged_content.json.gz data/taxons.json

data/content.json.gz:
	cd python && python3 -u -c "from data_extraction.export_data import export_content; export_content(output_filename='../data/content.json.gz')"

data/export_filtered_content.json.gz : data/content.json.gz
	cd python && python3 -u -c "from data_extraction.export_data import export_filtered_content; export_filtered_content(input_filename='../data/content.json.gz', output_filename='../data/filtered_content.json.gz')"

data/export_untagged_content.json.gz : data/content.json.gz
	cd python && python3 -u -c "from data_extraction.export_data import export_untagged_content; export_untagged_content(input_filename='../data/content.json.gz', output_filename='../data/untagged_content.json.gz')"

data/taxons.json.gz:
	cd python && python3 -u -c "from data_extraction.export_data import export_taxons; export_taxons(output_filename='../data/taxons.json.gz')"

$(DATADIR)/clean_taxons.csv.gz: $(DATADIR)/taxons.json.gz
	python3 python/clean_taxons.py $< $@

$(DATADIR)/clean_content.csv.gz $(DATADIR)/untagged_content.csv.gz : python/clean_content.py $(DATADIR)/content.json.gz \
    $(DATADIR)/document_type_group_lookup.json
	python3 python/clean_content.py

$(DATADIR)/new_content.csv.gz : python/create_new.py $(DATADIR)/untagged_content.csv.gz \
    $(DATADIR)/old_taxons.csv.gz
	python3 python/create_new.py

$(DATADIR)/labelled.csv.gz : python/create_labelled.py $(DATADIR)/clean_content.csv.gz \
    $(DATADIR)/clean_taxons.csv.gz
	python3 python/create_labelled.py

$(DATADIR)/document_type_group_lookup.json :
	aws s3 cp $(S3BUCKET)/document_type_group_lookup.json $(DATADIR)/document_type_group_lookup.json

$(DATADIR)/raw_taxons.json :
	aws s3 cp $(S3BUCKET)/raw_taxons.json $(DATADIR)/raw_taxons.json

$(DATADIR)/raw_content.json.gz :
	aws s3 cp $(S3BUCKET)/raw_content.json.gz $(DATADIR)/raw_content.json.gz

# Forgive this horrible repetition.

upload: labelled
	aws s3 cp $(DATADIR)/untagged_content.csv.gz $(S3BUCKET)/untagged_content.csv.gz
	aws s3 cp $(DATADIR)/empty_taxons.csv.gz $(S3BUCKET)/empty_taxons.csv.gz
	aws s3 cp $(DATADIR)/labelled.csv.gz $(S3BUCKET)/labelled.csv.gz
	aws s3 cp $(DATADIR)/filtered.csv.gz $(S3BUCKET)/filtered.csv.gz
	aws s3 cp $(DATADIR)/old_taxons.csv.gz $(S3BUCKET)/old_taxons.csv.gz
	aws s3 cp $(DATADIR)/labelled_level1.csv.gz $(S3BUCKET)/labelled_level1.csv.gz
	aws s3 cp $(DATADIR)/labelled_level2.csv.gz $(S3BUCKET)/labelled_level2.csv.gz
	aws s3 cp $(DATADIR)/empty_taxons_not_world.csv.gz $(S3BUCKET)/empty_taxons_not_world.csv.gz
	aws s3 cp $(DATADIR)/new_content.csv.gz $(S3BUCKET)/new_content.csv.gz


clean :
	-rm -f $(DATADIR)/clean_taxons.csv.gz $(DATADIR)/clean_content.csv.gz\
	    $(DATADIR)/untagged_content.csv.gz  $(DATADIR)/empty_taxons.csv.gz  \
	    $(DATADIR)/labelled.csv.gz  $(DATADIR)/filtered.csv.gz  $(DATADIR)/old_taxons.csv.gz  \
	    $(DATADIR)/labelled_level1.csv.gz  $(DATADIR)/labelled_level2.csv.gz  \
	    $(DATADIR)/empty_taxons_not_world.csv.gz  $(DATADIR)/new_content.csv.gz \
	    data/taxons.json data/content.json.gz data/export_untagged_content.json.gz data/export_filtered_content.json.gz

clean_all : clean
	-rm -f $(DATADIR)/document_type_group_lookup.json \
	    $(DATADIR)/raw_taxons.json $(DATADIR)/raw_content.json.gz

pip_install:
	pip3 install -r python/requirements.txt

check:
	cd python && python3 -m pytest

help :
	@cat Makefile

.PHONY : pip_install check clean clean_all upload help
