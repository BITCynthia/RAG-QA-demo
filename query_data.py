import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

from agent.embedding import OpenAIEmbedding
from agent.chat import OpenAIClient

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
[Begin of the question]{question}[End of the question]
[Begin of the context]{context}[End of the context]
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", type=str, help="Query Text")
    args = parser.parse_args()
    return args

def load_database(CHROMA_PATH):
    embedding_function = OpenAIEmbedding().get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db

def make_prompt(results, query_text):
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    return prompt

def get_response(prompt):
    model = OpenAIClient().get_model()
    response = model.predict(prompt)
    formatted_response = f"Response: {response}\n"
    return formatted_response

def main():
    CHROMA_PATH = "chroma"

    args = parse_args()
    query_text = args.query

    db = load_database(CHROMA_PATH)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0:
        print(f"Unable to find matching results.")
        return

    prompt = make_prompt(results, query_text)
    response = get_response(prompt)
    print(response)



if __name__ == "__main__":
    main()