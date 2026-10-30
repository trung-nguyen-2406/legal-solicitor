from apps.services.db_service import DatabaseService

db = DatabaseService()

res = db.hydrate_chunks([{"chunk_id": "0000402e-7a80-5526-84ef-8ed0efcab287"}])
print("--- HYDRATE ---")
print(res[0]["anchor_tag"])
print(res[0]["raw_content"])
print()

doc_id = res[0]["doc_id"]
doc_number = res[0]["doc_number"]
chunk_id = res[0]["chunk_id"]

pinned = db.fetch_amending_provisions(
    doc_ids=[doc_id],
    existing_chunk_ids={chunk_id},
    article_codes=["Điều 21"],
    limit=2,
)
print("--- AMENDING PROVISIONS ---")
print("count:", len(pinned))
for p in pinned:
    print(p["anchor_tag"], p["raw_content"][:100])
print()

expanded = db.expand_bidirectional(res, limit=3)
print("--- EXPAND BIDIRECTIONAL ---")
print("count:", len(expanded))
for e in expanded:
    print(e["anchor_tag"], e["raw_content"][:100])
print()

lineage = db.fetch_amendment_lineage_cte(doc_number, limit=5)
print("--- LINEAGE CTE ---")
print("count:", len(lineage))
for l in lineage:
    print(l["anchor_tag"], l["raw_content"][:100])