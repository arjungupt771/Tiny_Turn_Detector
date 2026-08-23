# from ai.llm_client import general_answer

# # print(general_answer(query="tell me the uploaded file is aligned to selected policy or not"))

# print(general_answer(query="hii"))

# from policy.manager import list_policies

# print(list_policies(include_disabled=True))

from storage.qdrant_client import get_qdrant_client
from qdrant_client.models import (
    PointStruct, Filter, FieldCondition, MatchValue, FilterSelector,)

from ai.llm_client import policy_alignment_data

from collections import defaultdict

client = get_qdrant_client()

UPLOADED_FILES = "uploaded_policies"

all_chunks = []
next_page = None

while True:
    points, next_page = client.scroll(
        collection_name=UPLOADED_FILES,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="file_name",
                    match=MatchValue(
                        value="product-1102-UN-System-Model-Policy-on-Sexual-Harassment-FINAL.pdf"
                    )
                )
            ]
        ),
        limit=100,
        with_payload=True,
        with_vectors=False,
        offset=next_page,
    )

    all_chunks.extend(points)

    if next_page is None:
        break

# Group by page
page_chunks = defaultdict(list)

for point in all_chunks:
    pages = point.payload.get("pages", [])

    for page in pages:
        page_chunks[page].append(point.payload)

# Print page-wise
policy_text = ""
for page in sorted(page_chunks.keys()):
    for chunk in page_chunks[page]:
        policy_text += "".join(chunk["text"])


data = policy_alignment_data(policy_text=policy_text)

print(data)