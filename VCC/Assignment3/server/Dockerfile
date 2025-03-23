FROM python:3.10

WORKDIR /app

COPY server.py.py /app

RUN pip install --no-cache-dir -r ../requirements.txt

EXPOSE 5050

CMD ["python", "server.py"]
