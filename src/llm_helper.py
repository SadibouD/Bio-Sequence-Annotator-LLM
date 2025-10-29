

print("Chargement llm_helper.py")
try:
    from openai import OpenAI
except Exception as e:
    print("Erreur d'import openai :", e)
    raise

#from openai import OpenAI

def annotate_protein(protein_seq):
    client = OpenAI()
    prompt = f"""Voici une séquence protéique : {protein_seq}
    Donne une description biologique possible de cette protéine,
    en te basant sur sa longueur, ses motifs, et ses similitudes connues.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

