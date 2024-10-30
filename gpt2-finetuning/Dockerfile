# Dockerfile
FROM python:3.8

# Nainstaluj potřebné Python balíčky
RUN pip install torch datasets transformers

# Zkopíruj skript do adresáře /app v kontejneru
COPY finetuning.py /app/finetuning.py

# Nastav pracovní adresář
WORKDIR /app

# Spusť příkaz při spuštění kontejneru
CMD ["python", "finetuning.py"]
