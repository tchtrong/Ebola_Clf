# %%

import subprocess

for i in range(3, 20):
    process = subprocess.run(["/home/binn/.meme/bin/meme",
                        #"-searchsize", "0",
                        "-protein",
                        "-objfun", "classic",
                        "-oc", "test_motifs/{}_{}".format(i, i),
                        "-mod", "zoops",
                        "-minw", str(i), "-maxw", str(i),
                        #"-nmotifs", "10",
                        "-minsites", "23",
                        "-evt", "0.05",
                        "-p", '10 --use-hwthread-cpus',
                        "fasta_cleaned/all.fasta"], capture_output=True)

# %%
