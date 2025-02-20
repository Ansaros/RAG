import os
import time
import logging
import numpy as np
from typing import List, Union
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("rag_system.log"), logging.StreamHandler()]
)

#Token usage count
token_usage = {"in": 0, "out": 0, "npz": 0}

def log_message(level: str, message: str):
    levels = {
        "debug": logging.debug,
        "info": logging.info,
        "warning": logging.warning,
        "error": logging.error
    }
    levels.get(level.lower(), logging.info)(message)

def extract_text_from_pdf(pdf_file: Union[str, bytes]) -> str:
    log_message("info", "Извлечение текста из PDF")
    try:
        reader = PdfReader(pdf_file)
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        log_message("debug", f"Извлечено {len(text)} символов из PDF.")
        return text
    except Exception as e:
        log_message("error", f"Не удалось извлечь текст из PDF: {e}")
        raise

def chunk_text_with_headers(text: str, max_length: int = 500) -> List[str]:
    log_message("info", "Разбиение текста на части...")
    words = text.split()
    chunks, current_chunk = [], []
    for word in words:
        current_chunk.append(word)
        if sum(len(w) + 1 for w in current_chunk) > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    log_message("debug", f"Текст разбит на {len(chunks)} частей.")
    return chunks

def get_embeddings(client: OpenAI, texts: List[str]) -> np.ndarray:
    log_message("info", f"Получение эмбеддингов для {len(texts)} текстов...")
    try:
        response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
        token_usage["in"] += response.usage.total_tokens
        embeddings = np.array([item.embedding for item in response.data])
        log_message("debug", f"Получено {len(embeddings)} эмбеддингов.")
        return embeddings
    except Exception as e:
        log_message("error", f"Ошибка получения эмбеддингов: {e}")
        raise

def save_embeddings_npz(chunks: List[str], embeddings: np.ndarray, output_file: str = "embedded_data.npz"):
    log_message("info", "Сохранение эмбеддингов в NPZ файл...")
    np.savez_compressed(output_file, chunks=chunks, embeddings=embeddings)
    token_usage["npz"] = os.path.getsize(output_file)
    log_message("info", f"Эмбеддинги сохранены в {output_file} ({token_usage['npz']} байт).")

def generate_answer(client: OpenAI, query: str, relevant_chunks: List[str]) -> str:
    log_message("info", "Генерация ответа...")
    context = "\n".join(relevant_chunks)
    messages = [
        {"role": "system", "content": "Вы — полезный ассистент компании. Используйте только информацию из базы знаний."},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}
    ]
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=2000)
        token_usage["in"] += response.usage.prompt_tokens
        token_usage["out"] += response.usage.completion_tokens
        return response.choices[0].message.content.strip()
    except Exception as e:
        log_message("error", f"Ошибка генерации ответа: {e}")
        raise

def rag_system(client: OpenAI, pdf_file: Union[str, bytes], query: str) -> str:
    log_message("info", "Запуск RAG системы...")
    try:
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text_with_headers(text)
        embeddings = get_embeddings(client, chunks)
        save_embeddings_npz(chunks, embeddings)
        relevant_chunks = chunks[:1]  # Упрощённый отбор топ-10 релевантных кусков
        answer = generate_answer(client, query, relevant_chunks)
        log_message("info", f"Токены — IN: {token_usage['in']}, OUT: {token_usage['out']}, NPZ: {token_usage['npz']} байт.")
        return answer
    except Exception as e:
        log_message("error", f"Сбой системы RAG: {e}")
        raise

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    pdf_path = "pdffiles/всяинф.pdf"
    query = "Лечение поверхностного кариеса кто делает"
    
    try:
        with open(pdf_path, "rb") as pdf_file:
            answer = rag_system(client, pdf_file, query)
            print("Ответ:", answer)
            print(f"Потраченные токены — IN: {token_usage['in']}, OUT: {token_usage['out']}, NPZ размер: {token_usage['npz']} токен.")
    except Exception as e:
        log_message("error", f"Ошибка обработки PDF: {e}")
