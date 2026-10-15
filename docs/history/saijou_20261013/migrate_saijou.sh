#!/usr/bin/env bash
# Migrate closed Saijou/WT1 artifacts from experiments/ism to /work3.
#
#   migrate_saijou.sh plan     # write list + sizes, no changes
#   migrate_saijou.sh copy     # rsync to /work3 (idempotent, resumable)
#   migrate_saijou.sh verify   # dry-run rsync must show no differences; counts/bytes equal
#   migrate_saijou.sh swap     # move originals to a holding dir, leave symlinks at old paths
#   migrate_saijou.sh unswap   # undo swap (restore originals from holding dir)
#
# Originals are never deleted by this script; the holding dir is removed by hand
# after acceptance.
set -euo pipefail

SRC=${SRC:-/work2/Users/qijun/gReLU-replication/experiments/ism}
DST=${DST:-/work3/Users/qijun/gReLU-history/saijou-20261013/ism}
HOLD=${HOLD:-/work2/Users/qijun/gReLU-replication/experiments/.pre-work3-20261015}
HERE=$(cd "$(dirname "$0")" && pwd)
LIST=$HERE/migrate.list
KEEP='^20261003_hsc_ag_training_data_similarity$'   # track_similarity: retained in place

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

make_list() {
  (cd "$SRC" && find . -mindepth 1 -maxdepth 1 ! -type l -printf '%f\n' | grep -Ev "$KEEP" | LC_ALL=C sort) > "$LIST"
}

stats() {  # <root> -> "files dirs links bytes" for every entry in LIST
  (cd "$1" && while read -r n; do
     find "$n" -printf '%y %s\n'
   done < "$LIST") | awk '{c[$1]++; if ($1=="f") b+=$2} END {printf "files=%d dirs=%d links=%d bytes=%d\n", c["f"], c["d"], c["l"], b}'
}

fix_links() {  # make absolute links into SRC relative, inside DST only
  find "$DST" -type l | while read -r l; do
    t=$(readlink "$l")
    case "$t" in
      "$SRC"/*)
        rel=$(realpath -m --relative-to="$(dirname "$l")" "$DST/${t#"$SRC"/}")
        ln -sfn "$rel" "$l"
        log "relinked ${l#"$DST"/} -> $rel" ;;
    esac
  done
}

verify() {
    # Dry run (size+mtime). Links differ by design after fix_links, so exclude them.
    diff=$(rsync -aHrn --files-from="$LIST" --no-links --itemize-changes "$SRC"/ "$DST"/ 2>&1 | grep -E '^([<>ch*]|rsync)' || true)
    s=$(stats "$SRC"); d=$(stats "$DST")
    log "src $s"; log "dst $d"
    broken=$(find "$DST" -xtype l | wc -l)
    if [ -z "$diff" ] && [ "$s" = "$d" ] && [ "$broken" -eq 0 ]; then
      log "VERIFY OK"
    else
      printf '%s\n' "$diff" | head -50; log "broken links in dst: $broken"; log "VERIFY FAILED"; return 1
    fi
}

swapped() { [ -d "$HOLD" ] && [ -n "$(ls -A "$HOLD")" ]; }

case "${1:-}" in
  plan)
    swapped && { echo "already swapped; unswap first" >&2; exit 1; }
    make_list
    log "$(wc -l < "$LIST") entries -> $DST"
    (cd "$SRC" && xargs -a "$LIST" -d '\n' du -sh --apparent-size -c) | tail -1
    log "src $(stats "$SRC")"
    ;;
  copy)
    [ -s "$LIST" ] || { echo "run plan first" >&2; exit 1; }
    swapped && { echo "already swapped; unswap first" >&2; exit 1; }
    mkdir -p "$DST"
    # -H keeps r4/r5 hardlinks; one call so links across entries are seen together.
    rsync -aHr --files-from="$LIST" --info=stats2 "$SRC"/ "$DST"/
    fix_links
    log "copy done"
    ;;
  verify) verify ;;
  swap)
    verify
    mkdir -p "$HOLD"
    while read -r n; do
      [ -L "$SRC/$n" ] && continue
      [ -e "$HOLD/$n" ] && { echo "holding exists: $n" >&2; exit 1; }
      mv "$SRC/$n" "$HOLD/$n"
      ln -s "$DST/$n" "$SRC/$n"
    done < "$LIST"
    log "swapped; originals in $HOLD"
    ;;
  unswap)
    while read -r n; do
      if [ -L "$SRC/$n" ] && [ -e "$HOLD/$n" ]; then rm "$SRC/$n"; mv "$HOLD/$n" "$SRC/$n"; fi
    done < "$LIST"
    log "restored originals"
    ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
