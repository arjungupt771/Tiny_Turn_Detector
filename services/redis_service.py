
import redis
import json
from collections import deque

# Redis connection
r = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

SESSION_TTL =  86400 #  24 hrs

MAX_PAIRS = 5
MAX_MESSAGES = 12 # MAX_PAIRS * 2   # only user + assistant (no system)

# ------------------ GET ------------------
def get_messages(session_id: str) -> deque:
    data = r.get(f"chat:{session_id}")

    if data:
        messages = json.loads(data)
        return deque(messages, maxlen=MAX_MESSAGES)

    return deque(maxlen=MAX_MESSAGES)


# ------------------ SAVE ------------------
def save_messages(session_id: str, messages: list):

    # print("=========================",type(messages))
    # 🔥 Remove system message if present
    if messages and messages[0].get("role") == "system":
        # messages = messages[1:]
        messages.popleft()

    # 🔥 Use deque to auto-trim
    dq = deque(messages, maxlen=MAX_MESSAGES)

    if len(dq) == 12:
        dq.popleft()
        dq.popleft()

    # Save to Redis
    r.setex(
        f"chat:{session_id}",
        SESSION_TTL,
        json.dumps(list(dq))   # convert back to list for JSON
    )


# ------------------ DELETE ------------------
def delete_session(session_id: str):
    r.delete(f"chat:{session_id}")