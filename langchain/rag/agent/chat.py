from tclogger import logger, Runtimer
from langchain_openai import AzureChatOpenAI

from constants import OPENAI_ENVS


class OpenAIClient:
    def __init__(self, openai_envs: dict = None):
        if not openai_envs:
            self.openai_envs = OPENAI_ENVS
        self.create_client()

    def create_client(self):
        if self.openai_envs.get("api_type") == "azure":
            self.client = AzureChatOpenAI(
                deployment_name = self.openai_envs.get("model"),
                azure_endpoint=self.openai_envs.get("endpoint"),
                api_key=self.openai_envs.get("api_key"),
                api_version=self.openai_envs.get("api_version"),
            )
        else:
            self.client = AzureChatOpenAI(
                base_url=self.openai_envs.get("endpoint"),
                api_key=self.openai_envs.get("api_key"),
            )

    def get_model(self):
        return self.client


if __name__ == "__main__":
    openai_client = OpenAIClient()
    messages = [
        {
            "role": "system",
            "content": "You are a greate benchmark engine named FLEX-RAG-Benchmarker.",
        },
        {
            "role": "user",
            "content": "what is your name?",
        },
    ]
    openai_client.create_response(messages=messages, stream=True)
    openai_client.display()

    # python -m agents.openai
