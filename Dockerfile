FROM python:3.6-stretch
ADD . .
RUN pip install -r requirements.txt
CMD ["python", "./bot.py"]