FROM python:3.7

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]