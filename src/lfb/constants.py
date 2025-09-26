AMINO_ACID_MAP = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY-")}
REVERSE_AMINO_ACID_MAP = {i: aa for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY-")}

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACID_TOKENS = list("ACDEFGHIKLMNPQRSTVWY-")


def is_standard_amino_acids(seq: str) -> bool:
    return all([aa.upper() in AMINO_ACIDS for aa in seq])
