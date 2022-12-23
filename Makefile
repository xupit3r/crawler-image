init:
	pip install -r requirements.txt

run:
	python3 run.py

test:
	nose2 tests
