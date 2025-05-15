FROM python:slim 

RUN pip install uv

WORKDIR /app 

COPY . . 

RUN uv pip install --no-cache --system -r requirements.lock


EXPOSE 5000

ENV FLASK_APP=src/api/api.py

CMD ["flask", "run", "--host", "0.0.0.0"]
