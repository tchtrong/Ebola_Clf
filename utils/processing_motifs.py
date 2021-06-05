from utils.processing_fimo import processing_motifs_fimo


def processing_motifs(no_X: bool, fimo: bool = True, length_range: range = range(3, 21)):
    if fimo:
        processing_motifs_fimo(no_X, length_range=length_range)
