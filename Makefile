# Makefile

data/clean_taxons.csv: python/clean_taxons.py
	python python/clean_taxons.py

#data/cleaned_content.json: python/clean_content.py
	#python python/clean_content.py

