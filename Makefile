install:
    pip install -r requirements.txt

freeze:
    pip freeze > requirements.txt

docker-build:
    docker build -t greek-llm .

docker-run:
    docker run -it greek-llm
