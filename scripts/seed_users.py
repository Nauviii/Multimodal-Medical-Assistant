"""One-off script to seed initial admin/doctor demo accounts."""

from scripts.db_session import _SessionLocal
from scripts.db_models import User
from api.middleware.auth import hash_password

DEMO_USERS = [
    {"username": "admin", "password": "changeme123", "role": "admin", "full_name": "Admin Demo"},
    {"username": "doctor", "password": "changeme123", "role": "doctor", "full_name": "Dr. Demo"},
]


def seed_users() -> None:
    """Insert demo admin and doctor accounts if they do not already exist."""
    db = _SessionLocal()
    try:
        for u in DEMO_USERS:
            if db.query(User).filter_by(username=u["username"]).first():
                continue
            db.add(User(
                username=u["username"],
                hashed_password=hash_password(u["password"]),
                role=u["role"],
                full_name=u["full_name"],
            ))
        db.commit()
        print("Seeded users:", [u["username"] for u in DEMO_USERS])
    finally:
        db.close()


if __name__ == "__main__":
    seed_users()