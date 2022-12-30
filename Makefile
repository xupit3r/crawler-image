init:
	pip install -r requirements.txt

train:
	TF_CPP_MIN_LOG_LEVEL=3 python3 image.py

predict:
	TF_CPP_MIN_LOG_LEVEL=3 python3 predict.py

summary:
	TF_CPP_MIN_LOG_LEVEL=3 python3 summary.py

test:
	nose2 tests
