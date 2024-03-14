FROM python:3.10-bookworm

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY api api
COPY setup.py setup.py

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
