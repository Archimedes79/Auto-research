for p in $(pgrep -f "pitch3/accept_chain.sh"); do kill $p; done
sleep 1
for p in $(pgrep -f "temp/accept_c4ec793f/bev_fusion"); do kill $p; done
pgrep -af "accept_chain|accept_c4ec793f" | grep -v pgrep | cut -c1-150
echo stopped
