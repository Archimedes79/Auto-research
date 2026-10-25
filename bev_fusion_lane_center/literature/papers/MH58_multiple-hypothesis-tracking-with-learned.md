# MH58 - Multiple hypothesis tracking with learned parts (online appearance models; neural gating of the hypothesis tree)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **A** (A abstract, M method, C code page). Verified: partly. Pipeline stage: S4, S8.*

## Papers

Multiple Hypothesis Tracking Revisited (Kim, Li, Ciptadi, Rehg, ICCV 2015) https://openaccess.thecvf.com/content_iccv_2015/html/Kim_Multiple_Hypothesis_Tracking_ICCV_2015_paper.html (PDF returned 403, listing and search summary only); Multi-object Tracking with Neural Gating Using Bilinear LSTM (Kim, Li, Rehg, ECCV 2018) https://openaccess.thecvf.com/content_ECCV_2018/html/Chanho_Kim_Multi-object_Tracking_with_ECCV_2018_paper.html (abstract and survey summary only)

## Idea

Track-oriented MHT keeps a tree of association hypotheses per track and prunes it; Kim 2015 adds online-trained regularised-least-squares appearance models per hypothesis branch; Kim 2018 uses a bilinear LSTM whose affinity score gates (prunes) branches of the hypothesis tree and scores tracks by appearance and motion.

## Formulation

not read. - input / output: detections per frame -> tracks.

## Training

Kim 2018: LSTM trained on tracklet appearance features; the tree itself is not trained.

## Differentiable?

no for the tree (gating network yes).

## Sensors and data

MOT16 / MOT17 (camera).

## Reported results

state-of-the-art near-online performance on MOT16 / MOT17 (abstract); numbers not seen. Searches for a learned or differentiable MHT with trainable hypothesis tree found none; transformer trackers (TrackFormer, MOTR, TADN) are not hypothesis trees.

## Compute

not stated.

## Code

none found

## Fit to our project

reuse_idea - effort M - confirms the pattern "learned gate decides which branch of a hypothesis tree survives" without end-to-end training of the tree.

## Needs

not read.

## Conflicts

not read.

## Tried in our experiments

none
