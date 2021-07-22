FROM python:3.7

RUN pip install numpy==1.18.1
RUN pip install torch==1.4.0

ADD requirements.txt /requirements.txt
RUN pip install -r requirements.txt

ADD main.py /main.py
ADD src /src

CMD [ "python", "main.py", "--query", "horse" ]