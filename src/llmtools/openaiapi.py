"""
Wrappers for OpenAI API.
"""

from typing import Any, List
import os
import sys

import openai as oai
import torch


EMBEDDING_DIM_DEFAULT = 1536
EMBEDDING_MODEL_DEFAULT = 'text-embedding-ada-002'


def set_api_key(api_key: str) -> None:
    """set the API key"""
    oai.api_key = api_key


def load_api_key(file_path: str) -> str:
    """load the API key from a text file"""
    if not os.path.isfile(file_path):
        print(f'OpenAI API key file `{file_path}` not found!')
        sys.exit()
    with open(file_path, 'r') as txt_file:
        res = txt_file.read().rstrip()
    if not res:
        print(f'Key file `{file_path}` empty!')
        sys.exit()

    return res


def get_models_list() -> List:
    """Return a list of available models."""
    model_obj= oai.Model.list()
    return [model.id for model in model_obj.data]


def embeddings(texts: List[str]) -> Any:
    """get embeddings, specifying the default model"""
    return oai.Embedding.create(
        input=texts,
        model=EMBEDDING_MODEL_DEFAULT
    )


def chat_completion(prompt: str, model: str) -> str:
    """simple chat completion"""
    res = oai.ChatCompletion.create(
        model=model,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
    )
    return res['choices'][0]['message']['content']


def extract_embeddings(data: Any) -> torch.Tensor:
    """extract embedings from an API response"""
    embs = [
        torch.tensor(x['embedding'])
        for x in data['data']]
    return torch.stack(embs, dim=0)


def embeddings_tensor(texts: List[str]) -> torch.Tensor:
    """get a tensor of embeddings from a list of strings"""
    embs = embeddings(texts)
    res = extract_embeddings(embs)
    assert res.shape == (len(texts), EMBEDDING_DIM_DEFAULT)
    return res
