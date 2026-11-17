p = "temp/day30/day30.sh"
s = open(p, encoding="utf-8").read()
s = s.replace("THREADS=${DAY30_THREADS:-1}", "THREADS=${DAY30_THREADS:-2}")
s = s.replace("# Threads of the dump: DAY30_THREADS (default 1: see the measurement in temp/e353/build_notes.md). The box is loaded by two trainings: nice 19, no timing claim. Never edit this script while it runs.",
              "# Threads of the dump: DAY30_THREADS (default 2: measured under load 1 thread = 1.6 - 1.9 s per frame, 2 threads = 0.9 - 1.3 s, temp/e353/threads_timing.log; the cards run on one). The box is loaded by two trainings: nice 19, no timing claim. Never edit this script while it runs.")
s = s.replace("(dry: ${DAY30_DRY:-no}, dump threads $THREADS)", "(dry: ${DAY30_DRY:-no}, dump threads $THREADS, timeout $TIMEOUT min)")
open(p, "w", encoding="utf-8", newline="\n").write(s)
