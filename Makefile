run:
	docker run test_langchain --gpus all --env-file .env

build:
	docker build -t test_langchain .