FROM python:3.8

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

RUN pip install pandas numpy scikit-learn streamlit

EXPOSE 8051

CMD ["streamlit", "run", "dep.py"]

