"""
Paths to data files and helper methods to load some of them.
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd


class _DataPaths():
    """
    A class for holding paths to data files.
    Do not reference directly. Use the global
    DATAPATHS variable instead.
    """

    DATA: Path = (Path(__file__) / ".." / "data").resolve()
    
DATAPATHS = _DataPaths()
"""Global object that holds paths."""

def __init__(
        self,
        trough: str,
        tmp: str):
    
    troughFolder="3D_Trough"+trough
    tmpFile = "tmp"+tmp+".txt"
    
    INSOLATION: Path = DATAPATHS.DATA / troughFolder / "insolation.txt"
    RETREAT: Path = DATAPATHS.DATA / troughFolder / "retreat_table.txt"
    OBLIQUITY: Path = DATAPATHS.DATA / troughFolder /  "obliquity.txt"
    ANGLE: Path = DATAPATHS.DATA /  troughFolder / "angle.txt"
    if tmp=='all':
        TMP: Path = DATAPATHS.DATA /  troughFolder / "TMPs/*" 
    else:
        TMP: Path = DATAPATHS.DATA /  troughFolder / "TMPs"/ tmpFile  


def load_retreat_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack the retreat data from the Bramson et al. thermal model used
    to create a bivariate spline. This data is 'static' and so can be
    loaded in here without respect to the model under consideration.

    Returns:
      times (np.ndarray): times the lags are measured at
      retreats (np.ndarray): retreat values in a 2D array of shape
        `(n_times, n_lags)`
      lags (np.ndarray): lag values the retreats have been calculated for
        by default these are [1,2,...15,20] in millimeters
    """
    df = pd.read_csv(DATAPATHS.RETREAT)
    times: np.ndarray = df["times"].values
    lag_cols: List[str] = [col for col in df.columns if col.startswith("lag")]
    lags: np.ndarray = np.array([int(col[3:]) for col in lag_cols])
    retreats: np.ndarray = df[lag_cols].values.T
        
    return times, retreats, lags


def load_obliquity_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Unpack the obliquity data.

    Returns:
      obliquity (np.ndarray): the obliquity values
      times (np.ndarray): times the obliquity is measured at
    """
    df = pd.read_csv(
        DATAPATHS.OBLIQUITY, names=["obliquity", "times"], skiprows=1, sep="\t"
    )
    times: np.ndarray = df["times"].values
    obl: np.ndarray = df["obliquity"].values
    return obl, times


def load_insolation_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Unpack the insolation data.

    Returns:
      insolation (np.ndarray): the insolation values
      times (np.ndarray): times the insolation is measured at
    """
        
    df = pd.read_csv(
    DATAPATHS.INSOLATION, names=["insolation", "times"], skiprows=1, sep="\t"
    )

    times: np.ndarray = df["times"].values
    ins: np.ndarray = df["insolation"].values
    return ins, times

def load_angle() -> Tuple[np.float]:
    """
    Loads the trough angle
    """
    angle=np.loadtxt(DATAPATHS.ANGLE)
    
    return angle


def load_TMP_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the TMP data for the trough being investigated now.

    Returns:
      x (np.ndarray): x position in kilometers
      y (np.ndarray): y position in meters
    """
    fnames=DATAPATHS.TMP
    tmpArraysList=[np.loadtxt(f, skiprows=1).T 
               for f in fnames]
    tmpArrays=np.concatenate(tmpArraysList)
    
    return tmpArrays


