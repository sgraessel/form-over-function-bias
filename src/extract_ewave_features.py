import polars as pl
from typing import NamedTuple


class Feature(NamedTuple):
    name: str
    description: str
    attestation: float
    example: str | None


EWAVE_PATH = "data/ewave"


def extract_top_n_features(language: str, nfeatures: int = 5) -> list[Feature]:
    langs = pl.read_csv(f"{EWAVE_PATH}/languages.csv")
    ids = langs.filter(pl.col("Name").str.to_lowercase() == language.lower())["ID"]
    if len(ids) != 1:
        raise ValueError(f"{len(ids)} matches for language {language!r}")

    values = pl.read_csv(
        f"{EWAVE_PATH}/values.csv", schema_overrides={"Example_ID": pl.String}
    )
    features = pl.read_csv(f"{EWAVE_PATH}/features.csv")
    examples = pl.read_csv(f"{EWAVE_PATH}/examples.csv")

    rows = (
        values.filter((pl.col("Language_ID") == ids[0]) & (pl.col("Value") == "A"))
        .select(
            "Parameter_ID",
            pl.col("Example_ID").str.extract(r"^(\d+)").cast(pl.Int64),
        )
        .join(
            features.select("ID", "Name", "Description", "Attestation"),
            left_on="Parameter_ID",
            right_on="ID",
        )
        .sort(
            ["Attestation", "Parameter_ID"], descending=[False, False], nulls_last=True
        )
        .head(nfeatures)
        .join(
            examples.select(pl.col("ID").alias("Example_ID"), "Primary_Text"),
            on="Example_ID",
            how="left",
        )
        .select("Name", "Description", "Attestation", "Primary_Text")
        .rows()
    )
    return [Feature(*r) for r in rows]