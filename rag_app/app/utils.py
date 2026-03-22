"""PDF text extraction utilities."""

from __future__ import annotations

from io import BytesIO
from typing import BinaryIO, Union

from pypdf import PdfReader


def extract_text_from_pdf(file: Union[BinaryIO, bytes, str]) -> str:
    """
    Extract plain text from a PDF.

    ``file`` may be a Streamlit UploadedFile, a file path, or raw bytes.
    """
    if isinstance(file, bytes):
        reader = PdfReader(BytesIO(file))
    elif hasattr(file, "read"):
        reader = PdfReader(file)
    else:
        reader = PdfReader(file)

    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()
