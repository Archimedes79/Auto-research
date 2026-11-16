import io
p = "temp/e363/start_chain.sh"
s = io.open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep("job() {   # job <id> <seed> <arm tag> <graph c2|a> <staged 0|1> <extra flags>\n  local id=$1 S=$2 tag=$3 g=$4 staged=$5 extra=$6", "job() {   # job <id> <seed> <arm tag> <graph c2|a> <staged 0|1> <spec> <extra flags>\n  local id=$1 S=$2 tag=$3 g=$4 staged=$5 spec=$6 extra=$7")
rep("--full $full --base $base --spec plain@0.5 $extra --tag $tag", "--full $full --base $base --spec $spec $extra --tag $tag")
i = s.index("for S in 0 1; do\n  PREV=\"\"")
j = s.index("done\nwait", i)
s = s[:i] + '''for S in 0 1; do
  PREV=""
  job E363.c_s$S  $S e363c  c2 0 plain@0.5 ""
  job E363.t_s$S  $S e363t  c2 0 close:1.5@0.5 ""
  job E363.a_s$S  $S e363a  a  0 plain@0.5 ""
  job E363.cw_s$S $S e363cw c2 1 plain@0.5 "--wire head --wire-run e358_w_s$S"
  job E363.aw_s$S $S e363aw a  1 plain@0.5 "--wire head --wire-run e358_w_s$S"
''' + s[j:]
s = s.replace("the eight CPU scoring jobs", "the ten CPU scoring jobs").replace("c -> a -> cw -> aw", "c -> t -> a -> cw -> aw")
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
