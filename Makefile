run:
	docker run --gpus all --env-file .env -p 8001:8001 -it testlangchain
build:
	docker build -t testlangchain .