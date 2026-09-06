from functools import cache

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

@cache
def _client() -> OpenAI:
    return OpenAI()

def model_call(system: str, user: str, model: str) -> str:
    """Call the OpenAI API to generate a response."""
    response = _client().responses.create(
        model=model,
        instructions=system,
        input=user,
    )
    return response.output_text

def parse_call[T: BaseModel](system: str, user: str, model: str, schema: type[T]) -> T:
    """Call the OpenAI API to provide a response in a specified schema."""
    response = _client().responses.parse(
        model=model,
        instructions=system,
        input=user,
        text_format=schema,
    )
    if response.output_parsed is None:
        raise ValueError(f"{model} returned no parsed output ...")
    return response.output_parsed

