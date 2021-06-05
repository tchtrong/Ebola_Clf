import subprocess
from pathlib import Path
import shutil
from utils.common import get_folder, DIR


def run_meme(no_X: bool = True, mode: str = "zoops", n_motifs: int = 20, length_range: range = range(3, 21), n_threads: int = 4, searchfull: bool = False, evt: float = 5.0e-002, minsites: int = 10):
    data = get_folder(DIR.DATA, no_X)
    motifs_folder = get_folder(DIR.MOTIFS, no_X)

    shutil.rmtree(motifs_folder, ignore_errors=True)
    motifs_folder.mkdir()
    data = data.joinpath("all.fasta")

    for i in length_range:
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
