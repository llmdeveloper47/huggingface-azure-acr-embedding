FROM python:3.8
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt


COPY . /app


# Get the models from Hugging Face to bake into the container
RUN python3 /app/app/model/model.py
        

EXPOSE 8000

ENTRYPOINT [ "uvicorn" ]

CMD [ "--host", "0.0.0.0", "app.main:app" ]
