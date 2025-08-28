import csv, os, pathlib
from collections import Counter, defaultdict

# === Pfade anpassen ===
INPUT  = r"C:\Users\rober\Desktop\football_match_predictor\data\raw\players_fbref_2000_2025.csv"
OUTDIR = r"C:\Users\rober\Desktop\football_match_predictor\data\interim"

# Die letzten 8 Felder sind fix:
TAIL = ["match_url","league","season","table_type","team_hint","match_title","league_slug","season_end"]

# Ab welcher Saison ist „neu“? (deine Beobachtung)
NEW_FROM = 2018  # d.h. 2017/18 -> season_end=2018

os.makedirs(OUTDIR, exist_ok=True)

def make_header(total_cols:int):
    lead = total_cols - 8
    generic = [f"c{str(i+1).zfill(2)}" for i in range(lead)]
    return generic + TAIL

def open_writer(tag:str, width:int, writers:dict):
    """tag in {'summary_alt','summary_neu','keeper_alt','keeper_neu'}"""
    fn = os.path.join(OUTDIR, f"players_{tag}.csv")
    # Falls schon ein Writer existiert, einfach wiederverwenden
    if tag in writers:
        return writers[tag], fn
    # Neu anlegen + Header passend zur Breite schreiben
    f = open(fn, "w", encoding="utf-8", newline="")
    w = csv.writer(f, lineterminator="\n")
    w.writerow(make_header(width))
    writers[tag] = (w, f, width)
    return writers[tag], fn

def main():
    widths = Counter()
    by_season_width = defaultdict(Counter)

    # --- Pass 1: Breiten grob scannen (nur Statistik/Info) ---
    with open(INPUT, "r", encoding="utf-8", newline="") as fin:
        rdr = csv.reader(fin)
        for i, row in enumerate(rdr):
            w = len(row)
            widths[w] += 1
            try:
                season_end = int(row[-1])
            except Exception:
                season_end = None
            if season_end:
                by_season_width[season_end][w] += 1
            if i > 100000:  # reicht als Stichprobe
                break

    print("Breiten (Stichprobe):", widths.most_common(5))
    print("Breite je season_end (Stichprobe):",
          {k: dict(v) for k, v in list(by_season_width.items())[:5]})

    # --- Pass 2: Splitten & schreiben ---
    writers = {}   # tag -> (writer, filehandle, width)
    counts  = Counter()

    with open(INPUT, "r", encoding="utf-8", newline="") as fin:
        rdr = csv.reader(fin)
        for row in rdr:
            w = len(row)
            if w < 8:
                continue  # korrupte Zeile

            table_type = (row[-5] or "").strip().lower()  # 'summary' oder 'keeper'
            try:
                season_end = int(row[-1])
            except Exception:
                season_end = None

            # primär über season_end entscheiden
            era = "neu" if (season_end is not None and season_end >= NEW_FROM) else "alt"

            # tag bestimmen
            if "keep" in table_type:
                tag = f"keeper_{era}"
            else:
                tag = f"summary_{era}"

            (wtr, fh, header_width), path = open_writer(tag, w, writers)

            # Falls erste Zeile eine andere Breite hat als Header → auf Headerbreite normalisieren
            if header_width != w:
                # pad/truncate
                if w < header_width:
                    row = row + [""] * (header_width - w)
                else:
                    row = row[:header_width]

            wtr.writerow(row)
            counts[tag] += 1

    # Close
    for wtr, fh, _ in writers.values():
        fh.close()

    print("\nFertig. Geschrieben:")
    for k, v in counts.items():
        print(f"  {k}: {v:,} Zeilen")
    print("Output-Ordner:", OUTDIR)

if __name__ == "__main__":
    main()
