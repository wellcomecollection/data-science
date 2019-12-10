FROM python:3.7

COPY requirements.txt requirements.txt
RUN pip install pip==9.0.3 pybind11
RUN pip install -r requirements.txt

COPY ./app /app

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]
