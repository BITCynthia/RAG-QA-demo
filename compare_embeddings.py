from langchain.evaluation import load_evaluator

from agent.embedding import OpenAIEmbedding

def main():
    embedding_function = OpenAIEmbedding().get_embeddings()
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apples: {vector}")
    print(f"Vector length: {len(vector)}")

    evaluator = load_evaluator(evaluator = "pairwise_embedding_distance", embeddings=embedding_function)
    words = ("apple", "apple")
    x = evaluator.evaluate_string_pairs(prediction = words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}: {x})")
    
if __name__ == "__main__":
    main()