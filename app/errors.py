class LLMError(RuntimeError):
    """The hosted LLM call failed or returned something unusable."""


class SuperheroError(RuntimeError):
    """The Superhero API call failed or returned a malformed payload."""


class SuperheroTimeout(SuperheroError):
    """The Superhero API did not respond within the configured timeout."""
