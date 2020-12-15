FROM python:3.8-slim-buster
RUN python -m pip install --upgrade pip && pip install pipenv
COPY Pipfile* ./
RUN pipenv install --system --deploy && python -m nltk.downloader punkt && python -m nltk.downloader averaged_perceptron_tagger
COPY ./app ./app
COPY .env .env
EXPOSE 8000
CMD uvicorn app.main:app