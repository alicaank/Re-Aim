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


class ProjectedGain(CumulativeGainRT):
 """Shared gain rule, projected form (arm correction_gainproj, written 2026-11-09 before any run with it; the same code
 for every model). Like CumulativeGainRT, but each feedback interval contributes only the part of the measured motion
 that goes along the commanded motion, so zero-mean measurement wobble cancels instead of adding up:
   translation: dc = R(C_prev)^T (c - c_prev), dm = R(M_prev)^T (m - m_prev)   (each in the camera's own frame at the
                interval start);  along = dm . dc/|dc|
   rotation:    wc = rotvec(R(C_prev)^T R(C)), wm = rotvec(R(M_prev)^T R(M))    (camera-frame turn, axis * angle);
                along = wm . wc/|wc|
   g_t = sum(along_t) / sum(|dc|),  g_r = sum(along_r) / sum(|wc|)   (running totals since the start)
 No window, no minimum, no clipping. A ratio keeps its previous value (start 1) when it is undefined: nothing commanded
 yet, a non-finite measurement, or a total along the command <= 0 (the model has not moved/turned the commanded way, so
 there is no positive gain to divide by). apply() is CumulativeGainRT.apply: translation offsets / g_t, rotation offsets
 (axis-angle, world frame, relative to the current command pose) / g_r. g_t = g_r = 1 is the original correction."""
 def update(self,command,measured):
  from scipy.spatial.transform import Rotation
  C=np.asarray(command,np.float64);M=np.asarray(measured,np.float64);Cp=self.cmd;Mp=self.meas
  dc=Cp[:3,:3].T@(C[:3,3]-Cp[:3,3]);dm=Mp[:3,:3].T@(M[:3,3]-Mp[:3,3])
  wc=Rotation.from_matrix(Cp[:3,:3].T@C[:3,:3]).as_rotvec();wm=Rotation.from_matrix(Mp[:3,:3].T@M[:3,:3]).as_rotvec()
  nc=float(np.linalg.norm(dc));nw=float(np.linalg.norm(wc))
  at=float(dm@dc/nc) if nc>0 else 0.0;ar=float(wm@wc/nw) if nw>0 else 0.0
  if np.isfinite(at) and np.isfinite(nc):
   self.CD+=nc;self.RD+=at
   if self.CD>0 and self.RD>0:self.g_t=self.RD/self.CD
  if np.isfinite(ar) and np.isfinite(nw):
   self.CA+=nw;self.RA+=ar
   if self.CA>0 and self.RA>0:self.g_r=self.RA/self.CA
  self.cmd=C.copy();self.meas=M.copy()
  return dict(g=self.g_t,g_rot=self.g_r,commanded_disp=nc,realized_disp=float(np.linalg.norm(dm)),realized_along=at,
              commanded_angle_deg=float(np.degrees(nw)),realized_angle_deg=float(np.degrees(np.linalg.norm(wm))),
              realized_along_angle_deg=float(np.degrees(ar)),total_commanded=self.CD,total_realized_along=self.RD,
              total_commanded_angle_deg=float(np.degrees(self.CA)),total_realized_along_angle_deg=float(np.degrees(self.RA)))


class SharedGainT(CumulativeGainRT):
 """Shared rule, movement only (arm correction_gaincum on models whose adapters take a 4x4 rule with apply(); written
 2026-11-10). Same g as CumulativeGain (total measured / total commanded camera-centre movement since the start, no
 limits); the turning ratio is kept at 1, so apply() changes only translation offsets, exactly as the CumulativeGain arms
 of Lyra-2 / SANA."""
 def update(self,command,measured):
  out=super().update(command,measured);self.g_r=1.0;out['g_rot']=1.0
  return out


class ProjectedGainFloor(ProjectedGain):
 """Projected rule with a floor of 1 (arm correction_gainprojf, written 2026-11-10 after the projected rule ran away on
 DreamX / WorldWarp / HY-WorldPlay / MatrixGame). Same totals as ProjectedGain, but the ratios used are max(1, ratio):
 the rule can only shrink the next commands (model moves/turns more than told), never enlarge them. Where a model moves
 less than told it is exactly the original correction. The raw ratios are logged as g_raw / g_rot_raw."""
 def update(self,command,measured):
  out=super().update(command,measured)
  raw_t=self.RD/self.CD if self.CD>0 else 1.0;raw_r=self.RA/self.CA if self.CA>0 else 1.0
  self.g_t=max(1.0,self.g_t);self.g_r=max(1.0,self.g_r)
  out.update(g=self.g_t,g_rot=self.g_r,g_raw=raw_t,g_rot_raw=raw_r)
  return out


# arm name -> rule class, for the adapters (all take (start_command, start_measured), update(command, measured), apply(commands, ref))
RULES={'correction_gaincum':SharedGainT,'correction_gainrt':CumulativeGainRT,'correction_gainproj':ProjectedGain,
       'correction_gainprojf':ProjectedGainFloor}
