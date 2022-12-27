init:
	pip install -r requirements.txt

run:
	TF_CPP_MIN_LOG_LEVEL=3 python3 run.py

test:
	nose2 tests
