FROM python:slim 

RUN pip install uv

WORKDIR /app 

COPY . .

RUN uv pip install --no-cache --system -r requirements.lock

EXPOSE 8080

ENV FLASK_APP=src/api/main.py 

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

CMD ["python", "src/api/main.py"]
