set -euo pipefail

HOTQA_DIR="dataset/hotqa"
ARCHIVE_ABSTRACT="$HOTQA_DIR/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
ARCHIVE_PROCESSED="$HOTQA_DIR/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"


usage() {
  echo "Usage: $0 [abstracts|processed]"
  echo "Default: processed"
}

mode="${1:-processed}"
case "$mode" in
  processed|p)
    ARCHIVE="$ARCHIVE_PROCESSED"
    ;;
  abstracts|abstract|a)
    ARCHIVE="$ARCHIVE_ABSTRACT"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    usage >&2
    exit 2
    ;;
esac

[[ -f "$ARCHIVE" ]] || { echo "Archive not found: $ARCHIVE" >&2; exit 1; }

echo "[archive] $ARCHIVE"


# 1) 解包到 HOTQA_DIR（archive 自带顶层目录）
pv "$ARCHIVE" | tar -xjf - -C "$HOTQA_DIR"

# 计算 TOP：去掉 .tar.bz2 后缀
NEWDIR="$(basename "$ARCHIVE")"
NEWDIR="${NEWDIR%.tar.bz2}"
TOP="$HOTQA_DIR/$NEWDIR"


total=$(find "$TOP" -type f -name '*.bz2' | wc -l | tr -d ' ')
find "$TOP" -type f -name '*.bz2' \
  | pv -l -s "$total" \
  | xargs -n 50 -P 8 bunzip2 -k

# 3) 删所有压缩包（.bz/.bz2）
find "$TOP" -type f \( -name '*.bz' -o -name '*.bz2' \) -delete

# 4) 每个语言子目录：wiki_* -> TOP/<lang>.jsonl
for d in "$TOP"/*/; do
	base="$(basename "$d")"

	mapfile -d '' files < <(
		find "$d" -maxdepth 1 -type f -name 'wiki_*' -print0 | sort -z -V
	)
	((${#files[@]})) || continue

	echo "[merge] $base: ${#files[@]} shards -> $TOP/${base}.jsonl"

	cat "${files[@]}" >"$TOP/${base}.jsonl"
	lines=$(wc -l <"$TOP/${base}.jsonl" | tr -d ' ')
	echo "[merge] $base done, lines=$lines"
done

# 5) 清理：只删除 TOP 下的一级子目录（保留 TOP/*.jsonl）
find "$TOP" -mindepth 1 -maxdepth 1 -type d -exec rm -rf -- {} +