# lit7_b: ADASIS map content and SD-map quality for an SD-map simulator

Tags: **[S]** primary source read; **[S2]** secondary or search summary; **[D]** derived by me from a sourced number; **[J]** my judgement.
Scope: only the ADASIS v2 spec (v2.0.1.0, Feb 2011) was read in full. v3 is member-only or delayed, so v3 rests on ERTICO/ADASIS releases, slides and a TomTom/Elektrobit whitepaper. WebFetch cached some PDFs automatically; I read them with pdftotext and downloaded nothing myself.

## Key findings
1. ADASIS v2 gives a **1-D path with offsets, not a lane map**: curvature/heading-change profiles, coarse lane counts, no lateral position, no lane geometry. Lane geometry, connectivity and markings arrive only with v3.
2. **ISO/TS 22726-1 is not the ADASIS Horizon**: its title is "Architecture and logical data model for harmonization of static map data". Your premise is probably wrong.
3. Navigation maps: "several meters" (TomTom); OSM about 6 m mean displacement (Haklay), 70% within 6 m (Cyprus); OSM lane count correct in at most about 80% (one city study).
4. SD-prior papers perturb with Gaussian shift sigma 0.25 to 2 m, rotation 2 to 10 deg, 10 to 30% polyline dropout. None models ADASIS directly.

## 1. ADASIS horizon content

| Item | v2 (2.0.1.0) [S] | v3 (3.x) [S2] |
|---|---|---|
| Structure | Paths (6-bit index, 8..63 real paths, 0..7 special), offset 13 bit = 1 m, mod 8191. Messages: POSITION, SEGMENT, STUB, PROFILE SHORT (two 10-bit spots, up to 1022 m apart), PROFILE LONG (32 bit), META-DATA. CAN. | Same path/profile idea over Ethernet; multiple providers; profile entries with explicit end offsets; connecting paths and MergePoint profile (v3.2). |
| Geometry | Absolute geometry is optional: LON/LAT LONG profiles (1e-7 deg), linear interpolation = polyline; altitude 0.01 m; Bezier control points. Spec says absolute geometry has "very limited usability" (6.6). Standard relative geometry: 10-bit nonlinear curvature (saturates at 0.162 1/m) and heading-change spots (360/254 = 1.417 deg steps). Spacing of spots is provider choice. | Road reference line (usually centreline) as polyline; heading, curvature, slope profiles; lane geometry for all lanes of a path section; linear objects (lane boundaries, markings, centre lines, curbs, guardrails). Resolution 0.01 m vs 1.0 m in v2.0.4. |
| Lanes | SEGMENT: lanes in driving direction 0..6 (6 means 6+), opposite direction 0..2 (2 means 2+), divided-road flag, form of way. POSITION: coarse current lane (emergency, single, left-most, right-most, middle). **No** width, markings or connectivity. | Lane model entries (new entry when lane count, connectivity or boundary type changes), lane numbering, lane connectivity profile, lane width (cm), localisation objects. |
| Position | Path index, offset (1 m), up to 4 position candidates, age (5 ms steps, max 2545 ms), speed (0.2 m/s), relative heading (1.417 deg), probability (5 bit), confidence (3 bit, implementation specific). **No lateral offset.** | Position messages with alternative positions and probabilities; "centimetre" claim is a product page (dSPACE). Fields unverified. |
| Horizon | Offset range 8190 m; cyclic offset allows unlimited length. Spec examples: 1 km simple client, 7 km high-end client. Event-based sending; message quota is system defined (example 10 msg/s); META-DATA about every 5 s (example). | "Long range horizon without restrictions" (2017 slide). No number found. |
| Branches | STUB message: sub-path index, turn angle, relative probability (5 bit, steps of 100/30 %, sums to 100), FRC, lane counts. Sub-path index 5 = stub without path data, 6 = continuation stub. KOM paper describes modes 0..3 (MPP only up to full network). | Tree of paths, most preferred path plus alternatives; MergePoint profile. |
| IDs | Link Identifier PROFILE LONG, 32 bit, e.g. GDF PVID, stable across releases (8.5). | LinkIdentifier profile. |

**ISO:** ISO/TS 22726-1 (2023, ed. 2 in 2025) = logical data model of *static map data* for connected and automated driving; Part 2 (2025) = dynamic data. A Technical Specification, not an ADASIS horizon. ADASIS-to-ISO link: not verified.
**Consequence for a simulator [J]:** a v2-like horizon yields a road centreline per carriageway, reconstructible from curvature/heading spots, with the ego snapped to it. Lateral ego position inside the road is unknown beyond the coarse lane flag.

## 2. SD-map geometric quality

| Quantity | Value | Source |
|---|---|---|
| Navigation/ADAS map accuracy | "several meters"; HD "few centimeters"; v2 describes a road as a single line | TomTom/Elektrobit whitepaper [S] |
| OSM vs Ordnance Survey (London/England, 2010) | about 6 m mean displacement; about 80% overlap for motorways | Haklay 2010 [S, abstract] |
| OSM Cyprus | about 70% of roads within 6 m of reference; better on higher road classes | Demetriou 2016 [S] |
| OSM Calgary / Ottawa / Vienna | mean offset 2.3 m; 76% within 10 m; 93% of reference length within 10 m (one district) | search summaries [S2] |
| OSM France | mean Hausdorff 13.57 m (preliminary poster) | [S2] |
| Consumer GNSS | 4.9 m radius, open sky (typical) | gps.gov [S] |
| HMM map-matching GPS noise | sigma 4.07 m (data recorded with Selective Availability on, so pessimistic) | Newson and Krumm 2009 [S] |
| Lane-count correctness in OSM | varies by city, "does not exceed 80%"; road type/direction above 90% to 95%; trunk entrance/exit lanes and turn restrictions above 85% | Lyft study via Gigazine [S2] |
| Road change rate | about 15% of roads per year (TomTom estimate); 5 to 20% per year (patents); INRIX keeps change below 5% per update | [S2] |
| Shape-point spacing | not found for commercial maps. OSM: a few m in curves, very long on straights; DP simplification is my proxy | [J] |

Unit conversion [D]: Cyprus 70% within 6 m corresponds to a 1-D Gaussian sigma of about 5.8 m. Haklay mean 6 m corresponds to sigma about 7.5 m. Calgary mean 2.3 m to sigma about 2.9 m. GPS 4.9 m (95% radius) to about 2 m per axis.

## 3. How academic SD-prior works perturb the prior

| Work | Prior and sampling | Perturbation actually used |
|---|---|---|
| SMERF (arXiv 2311.04079) | OSM polylines, N points evenly sampled, road-type one-hot | none; ego pose from GPS [S] |
| P-MapNet (2403.10521) | OSM raster, 0.15/0.3 m cells | none synthetic; misalignment handled by cross-attention; service roads removed (58.5 vs 60.2 mIoU) [S] |
| SMART (2502.04329) | OSM, 11 points per polyline, 100 x 50 m | no noise described [S] |
| TopoSD / Zhang et al. (2408.01471) | OSM raster or graph | translation 0.25, 0.5, 1, 2 m (random direction), rotation 0, 5, 10 deg; loss 3.1% at 1 m/0 deg, 5.0% at 1 m/5 deg, 21.1% at 2 m/10 deg [S] |
| SDTagNet (2506.08997) | OSM, 10 points per polyline | train augmentation sigma_trans 1 m, sigma_rot 2 deg (also "locally constant"), element drop about 0.1; did **not** help on clean validation (43.8/45.6 vs 48.1 mAP) [S] |
| PriorDrive (2409.05352) | OSM vectors | Gaussian sigma 1 m per element; pre-training noise on 5% of points, 10% of elements, N(0,1) [S] |
| NavMapFusion (2512.03317) | OSM, manually aligned | polyline dropout 0 to 100% (train 10 to 30%); translation sigma 0 to 3 m (train 1 m); random wrong location p [S] |
| Online Navigation Refinement (2507.07487) | OSM vs nuScenes HD | global shift 10/20/50% of map range, per-element jitter 5/10/20%, removal 10/20/30%; graceful degradation (val NR-F1 77.5 to 74.6 under shift) [S] |
| MapEx (2311.10517) | HD-derived priors | shift sigma 1 m per element; point noise sigma 5 m; outdated: 50% dividers deleted plus warping [S] |
| D2HDMap (2606.20725) | driveline prior | translation sigma 0 to 2 m, rotation sigma 0 to 5 deg [S] |

Pattern: nobody simulates ADASIS quantisation, lane-count errors or wrong connectivity. Typical stress ranges are 1 to 2 m, 2 to 5 deg, 10 to 30% drop.

## 4. Recommended parameter sets

Model [J]: per-road lateral bias b ~ N(0, sigma_b) plus a slowly varying component along arclength (correlation length 100 to 200 m); per-vertex jitter; global rigid registration error (ego pose vs map) applied separately from map shape error.

| Parameter | Good SD | ADASIS-v2-like | Poor SD | Basis |
|---|---|---|---|---|
| Lateral map bias sigma_b | 1.0 m | 2.5 m | 5.0 m | v2-like: "several meters" [S]; poor: Cyprus/Haklay [D]; good: NavMapFusion/PriorDrive train value 1 m [S]; choice [J] |
| Per-vertex jitter | 0.2 m | 0.5 m | 1.0 m | [J] (MapEx 5 m is an extreme stress test) |
| Shape-point spacing | 5 to 20 m, denser in curves | 10 to 50 m, 1 m offset quantisation, 1.417 deg heading steps | 20 to 100 m (DP tolerance about 3 m) | v2 quantisation [S]; spacing [J] |
| Longitudinal offset (map-matching) | sigma 1 m | sigma 3 m | sigma 8 m, junction jumps up to 20 m | gps.gov 4.9 m, Newson-Krumm 4.07 m [S]; scaling [J] |
| Heading noise | 0.5 deg | 2 deg | 5 deg | SDTagNet 2 deg, TopoSD 5 to 10 deg, D2HDMap 5 deg [S]; good [J] |
| Ego lateral registration | GNSS+DR sigma 0.5 m | 1.0 m plus unknown lane offset (no lateral position in v2) | 2.0 m | 4.9 m 95% radius [S]; v2 structure [S]; values [J] |
| Lane-count error rate | 5% | 10% (v2 field saturates at 6+) | 20 to 25% | Lyft at most 80% correct [S2]; others [J] |
| Missing-segment rate (minor roads first) | 2% | 5% | 15 to 20% | dropout 10 to 30% in NavMapFusion, Nav. Refinement [S]; rates [J] |
| Topology/outdated rate (wrong link, wrong turn, new road missing) | 2% | 5% | 15% | TomTom 15%/yr [S2], turn restrictions at least 85% correct [S2]; rates [J] |
| Lane geometry | none, one centreline per carriageway | same | same | v2 [S] |

Notes: values not tied to a cited study are my judgement. Calibrate against a real OSM-vs-HD comparison on your own data before trusting absolute numbers. SDTagNet found noise augmentation did not help on clean validation; evaluate on noisy validation too.

## Claims not verified
- ISO/TS 22726-1 as the ADASIS standard: search shows only a static-map logical model; any ADASIS link is unknown.
- ADASIS v3 field-level layout (position fields, horizon length, update rate, polyline spacing). The 0.01 m vs 1.0 m figure comes from the ERTICO/ADASIS announcement only.
- Production v2 horizon length and update rate: 1 to 8 km is spec capability plus examples, not measured practice. Later v2.0.2 to 2.0.5 changes not read.
- Calgary, Ottawa, Vienna, France figures: search summaries only. Lyft lane figure via Gigazine (Lyft blog returned 403).
- TomTom 15%/yr and patent 5 to 20%/yr change rates; vendor SD accuracy beyond "several meters"; commercial shape-point spacing; lane-count accuracy of commercial maps.
- Any paper simulating ADASIS horizon quality for ML: none found.
- D2HDMap, NavMapFusion, Nav. Refinement are recent preprints, not checked for peer review; the latter is inconsistent on removal rates (12% vs 20%).

## Sources
- ADASIS v2.0.1.0 spec: https://hvcs.ornl.gov/resources/docs/200v2.0.1-D2.2-ADASIS-v2-Specification.0.pdf
- ERTICO/ADASIS v3.1.0 release: https://adasis.org/2020/11/26/new-advanced-driver-assistance-systems-specification-provides-impactful-and-more-accurate-features-to-support-automated-driving/ ; https://erticonetwork.com/?p=131282
- SIP-ADUS slides 2017 (v3 lane model): https://www.sip-adus.go.jp/evt/workshop2017/file/evt_ws2017_s2_Jean-CharlesPandazis.pdf ; 2022: https://en.sip-adus.go.jp/evt/workshop2022/file/dm/4_DM_Pandazis-ADASIS-presentation-Final.pdf
- TomTom/Elektrobit whitepaper: https://download.tomtom.com/open/banners/Elektrobit_TomTom_whitepaper.pdf
- C2C-CC lane layouts (quotes ADASIS v3.1.0 lane model rules): https://www.car-2-car.org/fileadmin/documents/General_Documents/C2CCC_WP_2099_DescriptionOfLaneLayouts_V1.1.pdf
- Burgstahler et al., Horizon.KOM: https://www.kom.tu-darmstadt.de/papers/BPB16.pdf ; Kahl survey: https://arxiv.org/pdf/2104.01252 ; dSPACE v3: https://www.dspace.com/en/pub/home/products/sw/impsw/adasis_v3_hr_blockset.cfm
- ISO/TS 22726-1 and -2 (search results): https://www.iso.org/standard/88210.html ; https://www.iso.org/standard/78333.html
- Haklay 2010: https://ideas.repec.org/a/sae/envirb/v37y2010i4p682-703.html ; Cyprus: https://ktisis.cut.ac.cy/handle/20.500.14279/30425 ; GPS: https://archive.gps.gov/systems/gps/performance/accuracy/ ; Newson-Krumm: https://microsoft.com/en-us/research/wp-content/uploads/2016/12/map-matching-ACM-GIS-camera-ready.pdf ; Lyft summary: https://www.gigazine.net/gsc_news/en/20210210-openstreetmap-lyft
- Priors: SMERF https://arxiv.org/pdf/2311.04079 ; P-MapNet https://arxiv.org/html/2403.10521v2 ; SMART https://arxiv.org/html/2502.04329v1 ; TopoSD https://arxiv.org/html/2408.01471 ; SDTagNet https://arxiv.org/html/2506.08997v2 ; PriorDrive https://arxiv.org/html/2409.05352 ; NavMapFusion https://arxiv.org/html/2512.03317 ; Nav. Refinement https://arxiv.org/html/2507.07487 ; MapEx https://arxiv.org/html/2311.10517v2 ; D2HDMap https://arxiv.org/html/2606.20725v1 ; SD-map survey https://arxiv.org/pdf/2409.02415
