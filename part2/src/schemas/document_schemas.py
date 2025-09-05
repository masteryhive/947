"""Gotten from LangChain Core Module"""

from __future__ import annotations

import contextlib
import mimetypes
from collections.abc import Generator
from io import BufferedReader, BytesIO
from pathlib import PurePath
from typing import Any, Literal, Optional, Union, cast, Dict, List
from pydantic import ConfigDict, BaseModel, Field, field_validator, model_validator
from langchain_core.load.serializable import Serializable

PathLike = Union[str, PurePath]


class DocumentPayload(BaseModel):
    id: str
    user_id: str
    text: str
    document_type: str
    memory_type: str
    collection_name: str

class RAGPayload(BaseModel):
    id: str
    user_id: str
    text: str
    document_type: str
    memory_type: str
    collection_name: str
    metadata: Dict[str, Any]

class BaseMedia(Serializable):
    """Use to represent media content.

    Media objects can be used to represent raw data, such as text or binary data.

    LangChain Media objects allow associating metadata and an optional identifier
    with the content.

    The presence of an ID and metadata make it easier to store, index, and search
    over the content in a structured way.
    """

    # The ID field is optional at the moment.
    # It will likely become required in a future major release after
    # it has been adopted by enough vectorstore implementations.
    id: Optional[str] = None
    """An optional identifier for the document.

    Ideally this should be unique across the document collection and formatted
    as a UUID, but this will not be enforced.

    .. versionadded:: 0.2.11
    """

    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata associated with the content."""

    @field_validator("id", mode="before")
    def cast_id_to_str(cls, id_value: Any) -> Optional[str]:
        if id_value is not None:
            return str(id_value)
        else:
            return id_value

class Document(BaseMedia):
    """Class for storing a piece of text and associated metadata.

    Example:

        .. code-block:: python

            from langchain_core.documents import Document

            document = Document(
                page_content="Hello, world!",
                metadata={"source": "https://example.com"},
                score={"value": 0.653}
            )
    """

    page_content: str
    """String text."""
    score: Optional[dict] = None
    """An optional score dictionary associated with the document."""
    type: Literal["Document"] = "Document"

    def __init__(self, page_content: str, score: Optional[dict] = None, **kwargs: Any) -> None:
        """Pass page_content in as positional or named arg."""
        # my-py is complaining that page_content is not defined on the base class.
        # Here, we're relying on pydantic base class to handle the validation.
        # Exclude 'id' from the kwargs passed to the superclass
        kwargs.pop('id', None)
        super().__init__(page_content=page_content, score=score, **kwargs)  # type: ignore[call-arg]
        self.__dict__.update(kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "document"]

    def __str__(self) -> str:
        """Override __str__ to restrict it to page_content and metadata."""
        # The format matches pydantic format for __str__.
        #
        # The purpose of this change is to make sure that user code that
        # feeds Document objects directly into prompts remains unchanged
        # due to the addition of the id field (or any other fields in the future).
        #
        # This override will likely be removed in the future in favor of
        # a more general solution of formatting content directly inside the prompts.
        attributes = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k not in ['id', 'type']}
        return f"Document({attributes})"