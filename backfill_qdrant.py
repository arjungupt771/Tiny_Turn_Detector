# backfill_qdrant.py — run once from your project root
from policy import manager
from storage.uploader import save_policy
from storage.collections import create_collections

create_collections()  # no-op if it already exists

for meta in manager.list_policies(include_disabled=True):
    policy_id = meta["id"]
    try:
        requirements = manager.load_policy_requirements(policy_id)
        save_policy(policy_id, requirements)
        print(f"Backfilled '{policy_id}' ({meta.get('total_sections')} requirements)")
    except Exception as e:
        print(f"Failed to backfill '{policy_id}': {e}")