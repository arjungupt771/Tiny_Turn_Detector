import uuid

from qdrant_client.models import (
    PointStruct, Filter, FieldCondition, MatchValue, FilterSelector,
)

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
    OptimizersConfigDiff,
    
)
from collections import defaultdict
from storage.config import REQUIREMENTS_COLLECTION, UPLOADED_FILES,POLICY_INFO, POLICY_ALIGNED
from storage.retriever import policy_document_content
from storage.embeddings import embed_batch
from storage.qdrant_client import get_qdrant_client

from ai.llm_client import extract_policy_compliance_requirements

# Fixed namespace so the same (policy_id, category, requirement) always
# maps to the same point ID -- re-uploading a policy overwrites its
# points instead of duplicating them.
_NAMESPACE = uuid.UUID("f4b6c9a0-4b1e-4b8a-9e3f-2b7c6d5a1e00")


def _requirement_point_id(policy_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"policy_id:{policy_id}"))


# def save_policy(policy_id: str, requirements: dict) -> bool:
#     """
#     Embeds and upserts every individual requirement from a policy's
#     requirements.json into Qdrant. Nothing else (no PDF, no metadata
#     record) is stored -- those stay on local disk as-is.

#     requirements: { "category_name": ["requirement", ...], ... }
#     """
#     client = get_qdrant_client()

#     flat = [(cat, req) for cat, items in requirements.items() for req in items]
#     if not flat:
#         return True

#     vectors = embed_batch([req for _, req in flat])
#     points = [
#         PointStruct(
#             id=_requirement_point_id(policy_id, category, requirement),
#             vector=vector,
#             payload={
#                 "policy_id": policy_id,
#                 "category": category,
#                 "requirement": requirement,
#             },
#         )
#         for (category, requirement), vector in zip(flat, vectors)
#     ]

#     BATCH = 100
#     for i in range(0, len(points), BATCH):
#         client.upsert(collection_name=REQUIREMENTS_COLLECTION, points=points[i:i + BATCH])

#     return True


# def save_policy(policy_id: str, chunks:list) -> bool:
#     """
#     Embeds and upserts every individual requirement from a policy's
#     requirements.json into Qdrant. Nothing else (no PDF, no metadata
#     record) is stored -- those stay on local disk as-is.

#     requirements: { "category_name": ["requirement", ...], ... }
#     """
#     client = get_qdrant_client()

#     # flat = [(cat, req) for cat, items in requirements.items() for req in items]
#     # if not flat:
#     #     return True

#     vectors = embed_batch(texts=[i['text'] for i in chunks])

#     vector_dim = len(vectors[0])
#     if not client.collection_exists(UPLOADED_FILES):
#         print(f"Creating collection '{UPLOADED_FILES}' (dim={vector_dim})...")
#         client.create_collection(
#             collection_name=UPLOADED_FILES,
#             vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
#             hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
#             optimizers_config=OptimizersConfigDiff(indexing_threshold=20000)
#         )

#     points = [
#         PointStruct(
#             id= uuid.uuid4(),
#             vector=vector,
#             payload={
#                 "policy_id": policy_id,
#                 "text": meta['text'],
#                 "file_name": meta['metadata']['file_name'],
#                 "section_name": meta['metadata']['section_name'],
#                 "hierarchy": meta['metadata']['hierarchy'],
#                 "pages": meta['metadata']['pages'],
#                 },
#         )
#         for meta, vector in zip(chunks, vectors)
#     ]

#     BATCH = 100
#     for i in range(0, len(points), BATCH):
#         client.upsert(collection_name=UPLOADED_FILES, points=points[i:i + BATCH])

#     return True

def save_policy(policy_id: str, chunks: list) -> bool:
    """
    Embed and store policy chunks in Qdrant while preserving
    the original chunk order.
    """

    client = get_qdrant_client()

    if not chunks:
        return True

    # Generate embeddings
    vectors = embed_batch(
        texts=[item["text"] for item in chunks]
    )

    vector_dim = len(vectors[0])

    # Create collection if it doesn't exist
    if not client.collection_exists(UPLOADED_FILES):
        print(
            f"Creating collection '{UPLOADED_FILES}' "
            f"(dim={vector_dim})..."
        )

        client.create_collection(
            collection_name=UPLOADED_FILES,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000
            )
        )

    # Create points
    points = []

    for chunk_index, (meta, vector) in enumerate(zip(chunks, vectors)):

        points.append(
            PointStruct(
                id=uuid.uuid4(),
                vector=vector,
                payload={
                    "policy_id": policy_id,

                    # Original order
                    "chunk_index": chunk_index,

                    "text": meta["text"],
                    "file_name": meta["metadata"]["file_name"],
                    "section_name": meta["metadata"]["section_name"],
                    "hierarchy": meta["metadata"]["hierarchy"],
                    "pages": meta["metadata"]["pages"],
                },
            )
        )

    # Upsert in batches
    BATCH = 100

    for i in range(0, len(points), BATCH):
        client.upsert(
            collection_name=UPLOADED_FILES,
            points=points[i:i + BATCH]
        )

    return True

def policy_info(policy_id,region,version):

    client = get_qdrant_client()

    if not client.collection_exists(POLICY_INFO):
            print(f"Creating collection '{POLICY_INFO}'...")
            client.create_collection(
                collection_name=POLICY_INFO,
                vectors_config=VectorParams(
                    size=1,                 # Dummy vector
                    distance=Distance.COSINE)
                )

    client.upsert(
        collection_name=POLICY_INFO,
        points=[
            PointStruct(
                id=uuid.uuid4(),
                vector=[0.0],        # Ignore this
                payload={
                    "policy_id": policy_id,
                    "region": region,
                    "version": version,
                    "status": "selected"
                }
            )
        ]
    )

    return f"{policy_id} info is saved..."


# def old_policy_aliginment(policy_name:str):

#     print("x"*30)
#     print(policy_name)
#     print("x"*30)


#     client = get_qdrant_client()

#     # all_chunks = []
#     # next_page = None

#     # while True:
#     #     points, next_page = client.scroll(
#     #         collection_name=UPLOADED_FILES,
#     #         scroll_filter=Filter(
#     #             must=[
#     #                 FieldCondition(
#     #                     key="policy_id",
#     #                     match=MatchValue(
#     #                         value=policy_name
#     #                     )
#     #                 )
#     #             ]
#     #         ),
#     #         limit=100,
#     #         with_payload=True,
#     #         with_vectors=False,
#     #         offset=next_page,
#     #     )

#     #     all_chunks.extend(points)

#     #     if next_page is None:
#     #         break

#     # # Group by page
#     # page_chunks = defaultdict(list)

#     # for point in all_chunks:
#     #     pages = point.payload.get("pages", [])

#     #     for page in pages:
#     #         page_chunks[page].append(point.payload)

#     # # Print page-wise
#     # policy_text = ""
#     # for page in sorted(page_chunks.keys()):
#     #     for chunk in page_chunks[page]:
#     #         policy_text += "".join(chunk["text"])

#     policy_text = policy_document_content(policy_name=policy_name)

    
#     data = extract_policy_compliance_requirements(policy_text=policy_text)

#     print("+"*30)
#     print(f"before: {data}")
#     print("+"*30)

#     data = {"policy_name":policy_name, **data}

#     print("+"*30)
#     print(f"afer: {data}")
#     print("+"*30)


#     # print("x-"*30)
#     # print(len(data), data)
#     # print("x-"*30)

#     if not client.collection_exists(POLICY_ALIGNED):
#             print(f"Creating collection '{POLICY_ALIGNED}'...")
#             client.create_collection(
#                 collection_name=POLICY_ALIGNED,
#                 vectors_config=VectorParams(
#                     size=1,                 # Dummy vector
#                     distance=Distance.COSINE)
#                 )


#     client.upsert(
#         collection_name=POLICY_ALIGNED,
#         points=[
#             PointStruct(
#                 id=uuid.uuid4(),
#                 vector=[0.0],        # Ignore this
#                 payload=data
#             )
#         ]
#     )


#     return f"{POLICY_ALIGNED} info is saved..."


def policy_aliginment(policy_name:str):

    print("x"*30)
    print(policy_name)
    print("x"*30)


    client = get_qdrant_client()

   
    policy_text = policy_document_content(policy_id=policy_name)

    print("||"*30)
    print(len(policy_text))
    print("||"*30)

    # with open("./check_policy_text_retrival","wb") as file:
    #     file.writelines(policy_text)

    
    policy_compliance_requirements = extract_policy_compliance_requirements(policy_text=policy_text)

    # print("+"*30)
    # print(f"before: {policy_compliance_requirements}")
    # print("+"*30)

    # print("+"*30)
    # print(f"before: {type(policy_compliance_requirements)}")
    # print(f"before: {policy_compliance_requirements}")
    # print("+"*30)

    if "policy_name" not in policy_compliance_requirements.keys():
        policy_compliance_requirements = {"policy_name":policy_name, **policy_compliance_requirements}
    else:
        policy_compliance_requirements['policy_name'] = policy_name

    # print("+"*30)
    # print(f"afer: {policy_compliance_requirements}")
    # print("+"*30)


    # print("x-"*30)
    # print(len(data), data)
    # print("x-"*30)

    # POLICY_ALIGNED = "test_policy_aligned"
    if not client.collection_exists(POLICY_ALIGNED):
            print(f"Creating collection '{POLICY_ALIGNED}'...")
            client.create_collection(
                collection_name=POLICY_ALIGNED,
                vectors_config=VectorParams(
                    size=1,                 # Dummy vector
                    distance=Distance.COSINE)
                )


    client.upsert(
        collection_name=POLICY_ALIGNED,
        points=[
            PointStruct(
                id=uuid.uuid4(),
                vector=[0.0],        # Ignore this
                payload=policy_compliance_requirements
            )
        ]
    )


    return f"{POLICY_ALIGNED} info is saved..."


def test_policy_aliginment(policy_name:str):
    client = get_qdrant_client()

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
                            value=policy_name
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

    data = extract_policy_compliance_requirements(policy_text=policy_text)

    data = {"policy_name":policy_name, **data}

    # print("x-"*30)
    # print(len(data), data)
    # print("x-"*30)



    return data


def delete_policy(policy_id: str) -> None:
    """Removes every requirement point belonging to this policy."""
    client = get_qdrant_client()
    client.delete(
        collection_name=UPLOADED_FILES,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="policy_id", match=MatchValue(value=policy_id))])
        ),
    )

    # Delete policy metadata
    client.delete(
        collection_name=POLICY_INFO,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="policy_id", match=MatchValue(value=policy_id))])
        ),
    )

    # Delete policy metadata
    client.delete(
        collection_name=POLICY_ALIGNED,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="policy_name", match=MatchValue(value=policy_id))])
        ),
    )

