# %%
from Bio import Entrez
from pathlib import Path
import shutil

# %%
queries = {"Zaire": "zai.genbank", "Bombali": "bom.genbank", "Bundibugyo": "bun.genbank",
           "Tai Forest": "tai.genbank", "Reston": "res.genbank", "Sudan": "sud.genbank"}

Entrez.email = 'tchtrong@apcs.vn'
data_folder = Path("dataset_org")
shutil.rmtree(data_folder)
data_folder.mkdir()

for query, file in queries.items():
    handle = Entrez.esearch(db="protein", retmax="24000",
                            term="\"{} Ebolavirus\"[Organism] NOT UNVERIFIED".format(query))
    record = Entrez.read(handle)
    handle.close()
    handle = Entrez.efetch(
        db="protein", id=record['IdList'], rettype="gb", retmode="text")
    record = handle.read()
    with open(data_folder/file, 'w') as f:
        f.write(record)
