from .constants import OPENAI_EMBEDDING_ENVS

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

class OpenAIEmbedding:
    def __init__(self, openai_envs: dict = None):
        if not openai_envs:
            self.openai_envs = OPENAI_EMBEDDING_ENVS
        self.create_embeddings()

    def create_embeddings(self):
        if self.openai_envs.get("api_type") == "azure":
            self.embeddings = AzureOpenAIEmbeddings(
                model=self.openai_envs.get("model"),
                azure_endpoint=self.openai_envs.get("endpoint"),
                api_key=self.openai_envs.get("api_key"),
                openai_api_version=self.openai_envs.get("api_version"),
            )
        else:
            self.embeddings = OpenAIEmbeddings(
                model=self.openai_envs.get("model"),
                base_url=self.openai_envs.get("endpoint"),
                api_key=self.openai_envs.get("api_key"),
            )

    def get_embeddings(self):
        return self.embeddings