FROM python:3.9
COPY ./requirements.txt webapp/requirements.txt
COPY webapp/app.py /webapp
WORKDIR /webapp
COPY webapp/* /webapp
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY pytorch_model.bin /webapp
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
