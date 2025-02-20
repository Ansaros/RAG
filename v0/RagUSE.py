import os
import logging
import numpy as np
from typing import List, Union
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("rag_system.log"), logging.StreamHandler()]
)

token_usage = {
    "input_tokens": 0,  
    "output_tokens": 0,  
    "npz_bytes": 0  
}

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

def load_embeddings_npz(file_path: str = "embedded_data.npz"):
    log_message("info", "Загрузка эмбеддингов из NPZ файла...")
    try:
        data = np.load(file_path, allow_pickle=True)
        chunks = data["chunks"]
        embeddings = data["embeddings"]
        token_usage["npz_bytes"] = os.path.getsize(file_path)
        log_message("info", f"Загружены эмбеддинги из {file_path} ({token_usage['npz_bytes']} байт).")
        return chunks, embeddings
    except Exception as e:
        log_message("error", f"Ошибка загрузки NPZ: {e}")
        raise

def generate_answer(client: OpenAI, query: str, relevant_chunks: List[str]) -> str:
    log_message("info", "Генерация ответа...")
    context = "\n".join(relevant_chunks)
    messages = [
        {"role": "system", "content": "Вы — полезный ассистент компании. Используйте только информацию из базы знаний."},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=messages,
            max_tokens=3000
        )
        
        if response.usage:
            token_usage["input_tokens"] += response.usage.prompt_tokens
            token_usage["output_tokens"] += response.usage.completion_tokens
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        log_message("error", f"Ошибка генерации ответа: {e}")
        raise

def rag_system(client: OpenAI, query: str) -> str:
    log_message("info", "Запуск RAG системы...")
    try:
        chunks, embeddings = load_embeddings_npz()
        relevant_chunks = chunks[:5] 
        answer = generate_answer(client, query, relevant_chunks)
        log_message("info", 
            f"Использовано токенов - Входные: {token_usage['input_tokens']}, "
            f"Выходные: {token_usage['output_tokens']}, "
            f"Размер NPZ: {token_usage['npz_bytes']} байт.")
        return answer
    except Exception as e:
        log_message("error", f"Сбой системы RAG: {e}")
        raise

if __name__ == "__main__":
    
    api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)    
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY не найден в переменных окружения")
    
    query = "Лечение поверхностного кариеса кто делает на адресе туркестан"
    
    try:
        answer = rag_system(client, query)
        print("Ответ:", answer)
        print(
            f"Использование ресурсов:\n"
            f"- Входные токены: {token_usage['input_tokens']}\n"
            f"- Выходные токены: {token_usage['output_tokens']}\n"
            f"- Размер файла эмбеддингов: {token_usage['npz_bytes']} байт"
        )
    except Exception as e:
        log_message("error", f"Ошибка обработки запроса: {e}")
