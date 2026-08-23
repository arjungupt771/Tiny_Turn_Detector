

from qdrant_client.models import Filter, FieldCondition, MatchValue

from storage.config import REQUIREMENTS_COLLECTION, UPLOADED_FILES, POLICY_ALIGNED, POLICY_INFO
from storage.embeddings import embed
from storage.qdrant_client import get_qdrant_client


def search_requirements(query: str, top_k: int = 5, policy_id: str | None = None):
    client = get_qdrant_client()
    conditions = []
    if policy_id:
        conditions.append(FieldCondition(key="policy_id", match=MatchValue(value=policy_id)))

    results = client.query_points(
        collection_name=REQUIREMENTS_COLLECTION,
        query=embed(query),
        query_filter=Filter(must=conditions) if conditions else None,
        limit=top_k,
    )
    return [{"score": r.score, **r.payload} for r in results.points]


def get_requirements_for_policy(policy_id: str) -> list[dict]:
    client = get_qdrant_client()
    points, _ = client.scroll(
        collection_name=REQUIREMENTS_COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="policy_id", match=MatchValue(value=policy_id))]),
        limit=1000,
    )
    return [p.payload for p in points]



def querys_relevent_chunks(query:str):
    client = get_qdrant_client()

    query_vector = embed(query)

    results = client.query_points(
        collection_name="uploaded_policies",
        query=query_vector,
        limit=10,
    )

    data = [i.payload['text'] for i in results.points]

    return data 

def available_policies():
    client = get_qdrant_client()
    
    policy_ids = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=POLICY_INFO,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            policy_id = point.payload.get("policy_id")

            if policy_id:
                policy_ids.append(policy_id)

        if offset is None:
            break

    return sorted(set(policy_ids))

def policy_document_content(policy_id:str)->str:

    client = get_qdrant_client()

    all_chunks = []
    next_page = None

    while True:
        points, next_page = client.scroll(
            collection_name=UPLOADED_FILES,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="policy_id",
                        match=MatchValue(
                            value=policy_id
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


    # Restore original document order
    all_chunks.sort(
        key=lambda point: point.payload.get("chunk_index", 999999)
    )

    policy_text = ""

    for point in all_chunks:
        payload = point.payload

        policy_text += "".join(payload["text"])

    return policy_text



def policy_aligned_data(policy_name: str):
    client = get_qdrant_client()

    records, _ = client.scroll(
        collection_name=POLICY_ALIGNED,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="policy_name",
                    match=MatchValue(value=policy_name)
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not records:
        return None

    return records[0].payload

# def policy_aligned_data():

#     client = get_qdrant_client()

#     records, _ = client.scroll(
#         collection_name=POLICY_ALIGNED,
#         limit=100,
#         with_payload=True,
#         with_vectors=False)

#     payloads = [record.payload for record in records]

#     return payloads[0]


