import os
import argparse

from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

from agent.chat import OpenAIChat
from agent.embedding import OpenAIEmbedding

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", type=str, help="Query Text")
    args = parser.parse_args()
    return args

def load_chat_model():
    chat = OpenAIChat().get_chat_model()
    return chat

def load_database(chroma_path):
    embedding_function = OpenAIEmbedding().get_embeddings()
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_function
    )
    return db

def query_by_rag(model, db, query_text, top_k = 3):
    results = db.similarity_search_with_score(query_text, k=top_k)
    prompt = make_prompt(query_text, results)
    response = model.invoke(prompt)
    return response

def make_prompt(query_text, results):
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    [Begin of the question]{question}[End of the question]
    [Begin of the context]{context}[End of the context]
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = prompt_template.format(context=context_text, question=query_text)
    return prompt

def main():
    db = load_database(chroma_path)
    model = load_chat_model()

    args = parse_args()
    query_text = args.query
    response=query_by_rag(model, db, query_text, top_k=3)
    print(response.content)

if __name__ == "__main__":
    chroma_path = os.path.abspath('rag-pipeline/chroma')
    main()