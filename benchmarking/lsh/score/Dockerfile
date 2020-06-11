FROM python:3.7

WORKDIR /opt

COPY requirements.txt /opt/requirements.txt
RUN pip install -r requirements.txt

COPY . /opt

CMD ["python", "score_table.py"]
