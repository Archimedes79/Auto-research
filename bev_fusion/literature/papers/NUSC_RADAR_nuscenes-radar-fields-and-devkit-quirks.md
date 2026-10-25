# NUSC_RADAR nuScenes radar point fields and devkit quirks (data_classes.py RadarPointCloud)

| | |
|---|---|
| source | https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/utils/data_classes.py (devkit source, read via a summariser); schema_nuscenes.md does not define the radar fields |
| fact status | checked (source page read, 2026-10-25); exact values should be re-read in the local devkit before use |
| groups | F4: 3, A9: 1 |
| code / licence | nuscenes-devkit, Apache-2.0 (memory, not checked) |
| verdict | use (data facts for the radar encoder; not a paper) |

**What it is**: the point fields and default filters of the nuScenes radar loader, with the consequences for an encoder. Fields (18): x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, is_quality_valid, ambig_state, x_rms, y_rms, invalid_state, pdh0, vx_rms, vy_rms. x points forward, y left.

**Facts that matter for encoding**
- Position uncertainty is delivered by the sensor: x_rms and y_rms are fields of every point (checked: the names are in the header; that they are RMS position-error estimates is memory, as is the unit); none of the papers above uses them (their encoders take x, y only). Likewise vx_rms, vy_rms for velocity.
- Doppler: vx, vy are the radial velocity resolved on x / y; vx_comp, vy_comp are the ego-motion-compensated version, and the devkit docstring recommends the compensated values. CenterFusion, CRAFT and RCTrans use compensated velocity; CRN, RCBEVDet and MVFusion do not say.
- Default filters: invalid_states = [0] (valid cluster only), ambig_states = [3] (unambiguous Doppler only), dynprop_states = all 0-6. Meanings: ambig 0 invalid, 1 ambiguous, 2 staggered ramp, 3 unambiguous, 4 stationary candidates; dynProp 0 moving, 1 stationary, 2 oncoming, 3 stationary candidate, 4 unknown, 5 crossing stationary, 6 crossing moving, 7 stopped. Invalid codes include low RCS (0x01), near-field artefact (0x02), outside field of view (0x07). The default ambiguity filter removes 3 of 5 states, i.e. most static candidates; a lane-side-structure encoder may want ambig 3 and 4 and the dynProp channel as a feature.
- Multi-sweep merge (from_file_multisweep, default 5 sweeps, drops points within 1.0 m of the sensor): only x, y, z are transformed to the reference frame; vx, vy (and the compensated pair) are NOT rotated, and the time lag is a separate array, not a point dimension. For several radars with different mounting yaw (our case if we use front + front-left / right) the velocity vectors must be rotated by the sensor extrinsic and the ego-yaw change before stacking, or the vector channels are inconsistent. For a single front radar the error is the ego yaw change over the sweep window only.
- Sensor direction (mounting pose) is not a point field: it enters only through the extrinsics used for the xyz transform. None of the encoders reviewed here feeds the sensor id / azimuth to the network except MVFusion's per-camera radar view.

**Transferable technique**: a point feature vector of [x, y, RCS, vx_comp, vy_comp (rotated), t_lag, x_rms, y_rms, ambig / dynProp flags] with the footprint taken from the rms fields (instead of from RCS as in RCBEVDet).

**Cost and fit**: free (loader flags). Check: the local devkit version and the real value ranges of x_rms / y_rms on CAM_FRONT-side radars before building anything.

**Doubts**: second-hand reading of the source through a summariser; the devkit does not say in which frame vx, vy are expressed (sensor frame is the usual reading, memory).
