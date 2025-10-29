print("Chargement dna_tools.py")
try:
    from Bio.Seq import Seq
except Exception as e:
    print("Erreur d'import Bio.Seq :", e)
    raise

#from Bio.Seq import Seq

def find_orfs(seq, min_length=30):
    """Trouve les ORFs (open reading frames) dans une séquence ADN."""
    orfs = []
    dna = Seq(seq)
    rna = dna.transcribe()
    for frame in range(3):  # 3 cadres de lecture
        protein = rna[frame:].translate(to_stop=False)
        current_orf = []
        for i, aa in enumerate(protein):
            if aa == "M":  # start codon (AUG → M)
                current_orf = ["M"]
            elif current_orf:
                current_orf.append(aa)
                if aa == "*":  # stop codon
                    if len(current_orf) >= min_length:
                        orfs.append("".join(current_orf))
                    current_orf = []
    return orfs
