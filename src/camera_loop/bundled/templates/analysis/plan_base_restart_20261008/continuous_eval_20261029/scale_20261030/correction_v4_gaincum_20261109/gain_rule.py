"""Shared gain rule for every model adapter (correction_v4_gaincum). Imported by snapshots/lyra_plugin.py and
snapshots/sana_run.py through sys.path entries added by lyra_plugin_v3.py / sana_v3.py."""
import numpy as np
class CumulativeGain:
 """Shared gain rule (correction_v4_gaincum, fixed 2026-11-09 before any run with it; the same code for every model).
 At every feedback point: g = (total measured camera-centre movement since the start) / (total commanded camera-centre
 movement since the start), each a sum of per-interval displacements in the model's native units. No window, no minimum,
 no clipping. Only when g is undefined (nothing commanded or nothing measured yet, or a non-finite measurement) it keeps
 its previous value, which starts at 1 (= the original correction). The caller divides the commanded translation
 offsets of the next chunk by g; rotations are unchanged."""
 def __init__(self,start_command,start_measured):
  self.cmd=np.asarray(start_command,np.float64)[:3].copy();self.meas=np.asarray(start_measured,np.float64)[:3].copy()
  self.CD=0.0;self.RD=0.0;self.g=1.0
 def update(self,command,measured):
  command=np.asarray(command,np.float64)[:3];measured=np.asarray(measured,np.float64)[:3]
  cd=float(np.linalg.norm(command-self.cmd));rd=float(np.linalg.norm(measured-self.meas))
  if np.isfinite(cd) and np.isfinite(rd):
   self.CD+=cd;self.RD+=rd
   if self.CD>0 and self.RD>0:self.g=self.RD/self.CD
  self.cmd=command.copy();self.meas=measured.copy()
  return dict(g=self.g,commanded_disp=cd,realized_disp=rd,total_commanded=self.CD,total_realized=self.RD)



def _angle(Ra, Rb):
 """rotation angle (radians) between two 3x3 rotations"""
 c=(np.trace(np.asarray(Ra,np.float64)@np.asarray(Rb,np.float64).T)-1)/2
 return float(np.arccos(np.clip(c,-1.0,1.0)))


class CumulativeGainRT:
 """Shared gain rule with a rotation ratio (arm correction_gainrt, written 2026-11-09 before any run with it; the same
 code for every model). Two ratios, both built like CumulativeGain:
   g_t = total measured camera-centre movement since the start / total commanded movement since the start
   g_r = total measured camera rotation angle since the start / total commanded rotation angle since the start
 each a sum over feedback intervals (displacement / relative rotation angle between consecutive feedback points). No
 window, no minimum, no clipping; a ratio keeps its previous value (start 1) only when it is undefined (nothing
 commanded or measured yet, non-finite). apply() divides the next chunk's commands relative to the current command pose:
 translation offsets by g_t, rotation offsets (axis-angle of R_i R_ref^T, world frame) by g_r. g_t = g_r = 1 leaves the
 original correction unchanged. Poses are 4x4 camera-to-world."""
 def __init__(self,start_command,start_measured):
  self.cmd=np.asarray(start_command,np.float64).copy();self.meas=np.asarray(start_measured,np.float64).copy()
  self.CD=self.RD=self.CA=self.RA=0.0;self.g_t=1.0;self.g_r=1.0
 def update(self,command,measured):
  command=np.asarray(command,np.float64);measured=np.asarray(measured,np.float64)
  cd=float(np.linalg.norm(command[:3,3]-self.cmd[:3,3]));rd=float(np.linalg.norm(measured[:3,3]-self.meas[:3,3]))
  ca=_angle(command[:3,:3],self.cmd[:3,:3]);ra=_angle(measured[:3,:3],self.meas[:3,:3])
  if np.isfinite(cd) and np.isfinite(rd):
   self.CD+=cd;self.RD+=rd
   if self.CD>0 and self.RD>0:self.g_t=self.RD/self.CD
  if np.isfinite(ca) and np.isfinite(ra):
   self.CA+=ca;self.RA+=ra
   if self.CA>0 and self.RA>0:self.g_r=self.RA/self.CA
  self.cmd=command.copy();self.meas=measured.copy()
  return dict(g=self.g_t,g_rot=self.g_r,commanded_disp=cd,realized_disp=rd,commanded_angle_deg=float(np.degrees(ca)),
              realized_angle_deg=float(np.degrees(ra)),total_commanded=self.CD,total_realized=self.RD,
              total_commanded_angle_deg=float(np.degrees(self.CA)),total_realized_angle_deg=float(np.degrees(self.RA)))
 def apply(self,commands,ref):
  from scipy.spatial.transform import Rotation
  out=np.asarray(commands,np.float64).copy();ref=np.asarray(ref,np.float64)
  out[:,:3,3]=ref[:3,3]+(out[:,:3,3]-ref[:3,3])/self.g_t
  rel=Rotation.from_matrix(out[:,:3,:3]@ref[:3,:3].T)
  out[:,:3,:3]=Rotation.from_rotvec(rel.as_rotvec()/self.g_r).as_matrix()@ref[:3,:3]
  return out
