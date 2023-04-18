FROM python:3.10

WORKDIR /code

RUN pip install --upgrade pip
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

RUN mkdir -p /data

COPY ./api/app /code/app
COPY ./src /code/app/src

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
