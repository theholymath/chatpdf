FROM python:3.11.3-slim-bullseye
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pdfchat-st.py app.py
RUN mkdir -p .streamlit
CMD ["streamlit", "run", "app.py"]
