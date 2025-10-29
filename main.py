
# from Bio_annotateur.src.dna_tools import find_orfs
# from Bio_annotateur.src.llm_helper import annotate_protein

# # ...le reste du code...

from .src.dna_tools import find_orfs
from .src.llm_helper import annotate_protein

# import sys
# import os

# # 🔧 Forcer Python à trouver le dossier src
# sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


if __name__ == "__main__":
    # Exemple simple
    seq = "ATGGTTTACGTTTAAATGA"  # mini séquence test

    print("🔬 Séquence ADN :", seq)
    orfs = find_orfs(seq)
    print("ORFs trouvés :", orfs)

    if orfs:
        annotation = annotate_protein(orfs[0])
        print("\n🧾 Annotation LLM :")
        print(annotation)
