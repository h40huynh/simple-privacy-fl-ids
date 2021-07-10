from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class TrainProfile(db.Model):
    __tablename__ = "train_profile"

    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # Atrribute
    index = db.Column(db.Integer, default=-1)
    n_client = db.Column(db.Integer, default=2)
    n_round = db.Column(db.Integer, default=2)

    def __init__(self, n_client=None, n_round=None) -> None:
        self.n_client = n_client
        self.n_round = n_round


class AggregateDetail(db.Model):
    __tablename__ = "aggregate_detail"

    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # Attribute
    profile_id = db.Column(db.Integer, db.ForeignKey("train_profile.id"))
    round_id = db.Column(db.Integer)
    client_id = db.Column(db.Integer)
    n_sample = db.Column(db.Integer)
    # is_get_weight = db.Column(db.Boolean)

    def __init__(
        self, profile_id=None, round_id=None, client_id=None, n_sample=None
    ) -> None:
        self.profile_id = profile_id
        self.round_id = round_id
        self.client_id = client_id
        self.n_sample = n_sample
