FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install --yes python3 python3-pip && \
    apt-get clean

COPY requirements.txt /tmp/
RUN pip3 install --upgrade pip
RUN pip3 install -r /tmp/requirements.txt

RUN python3 -m nltk.downloader punkt

ADD data /data
ADD nerd /nerd
ADD ./run_ecs.py ./run_ecs.py

EXPOSE 80

CMD ["python3", "run_ecs.py"]
