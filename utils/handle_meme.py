import subprocess
from pathlib import Path
import shutil


def run_meme(data: Path, mode: str = "zoops", n_motifs: int = 20, n_threads: int = 4, searchfull: bool = False, evt: float = 5.0e-002, minsites: int = 10):
    data = Path(data)
    motifs_folder = "motifs"

    if "X" in data.name:
        motifs_folder += "_no_X"
    if mode == "oops":
        motifs_folder += "_oops"
    if mode == "anr":
        motifs_folder += "_anr"

    motifs_folder = Path(motifs_folder)
    
    shutil.rmtree(motifs_folder, ignore_errors=True)
    motifs_folder.mkdir()
    data = data.joinpath("all.fasta")

    for i in range(3, 21):
        sub_folder = motifs_folder.joinpath("{}".format(i))
        sub_folder.mkdir()

        args = ["meme",
                "-protein",
                "-objfun", "classic",
                "-oc", sub_folder.as_posix(),
                "-mod", mode,
                "-minw", str(i), "-maxw", str(i),
                "-nmotifs", str(n_motifs),
                "-minsites", str(minsites),
                "-evt", str(evt),
                "-p", '{} --use-hwthread-cpus'.format(
                    n_threads),
                data.as_posix()]
        if searchfull:
            args += ["-searchsize", "0"]
            
        process = subprocess.run(args, capture_output=True)
        if process.returncode != 0:
            print(process.stderr)
            break
        else:
            print("Finished finding motifs with length {}".format(i))
