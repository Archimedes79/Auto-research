import sys, time
sys.argv += ["--data", "nuscenes", "--sharp", "--pyr", "--doppler", "--rawwarp", "3.0", "--prior", "0.03", "--smooth", "0.05"]
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_explains as T
T.DEV = "cpu"
t0 = time.time()
seqs = T.nuscenes_sequences(True, 2, 5)
print(len(seqs), "sequences, maps", tuple(seqs[0]["maps"].shape), "steps", [len(s) for s in seqs[0]["steps"]], f"({time.time() - t0:.0f} s)")
T.sequence_report(T.Model().eval(), seqs, "temp/checks/dry_sequences.png")
