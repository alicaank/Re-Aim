# Rectified Gain Compensation

The native adapters use `ProjectedGainFloor` from `gain_rule.py`. It accumulates translation and rotation projected along commanded motion and floors each divisor at one.
