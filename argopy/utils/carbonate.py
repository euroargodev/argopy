"""
Utility module for carbonate calculations used in the `carbonate_content` extension.
"""

import numpy as np
import logging
from typing import Dict, List, Union
from dataclasses import dataclass
import PyCO2SYS as pyco2

log = logging.getLogger("argopy.utils.carbonate")

def calculate_uncertainties(
        deriv: Dict[str, np.ndarray],
        output_vars: List[str],
        PAR1: Union[float, np.ndarray],
        PAR2: Union[float, np.ndarray],
        PAR1TYPE: int,
        PAR2TYPE: int,
        ePAR1: Union[float, np.ndarray],
        ePAR2: Union[float, np.ndarray],
        eSAL: Union[float, np.ndarray],
        eTEMP: Union[float, np.ndarray],
        SI: Union[float, np.ndarray],
        eSI: Union[float, np.ndarray],
        PO4: Union[float, np.ndarray],
        ePO4: Union[float, np.ndarray],
        epK: Union[List[float], np.ndarray],
        eBt: float,
        r: float
    ) -> Dict[str, np.ndarray]:
        """
        Calculate total uncertainties for carbonate system variables using error propagation.
        Script largely adapted from PyCO2SYS errors.m function [5]_.
        
        Parameters
        ----------
        deriv : dict
            Dictionary containing partial derivatives for each output variable
        output_vars : list of str
            List of output variable names to calculate uncertainties for
        PAR1 : float or ndarray
            First parameter of the carbonate system
        PAR2 : float or ndarray
            Second parameter of the carbonate system
        ePAR1 : float or ndarray
            Uncertainty in PAR1
        ePAR2 : float or ndarray
            Uncertainty in PAR2
        eSAL : float or ndarray
            Uncertainty in salinity
        eTEMP : float or ndarray
            Uncertainty in temperature
        SI : float or ndarray
            Silicate concentration
        eSI : float or ndarray
            Uncertainty in silicate
        PO4 : float or ndarray
            Phosphate concentration
        ePO4 : float or ndarray
            Uncertainty in phosphate
        epK : list or ndarray
            Uncertainties in dissociation constants [pK0, pK1, pK2, pKb, pKw, pKspa, pKspc]
        eBt : float
            Fractional uncertainty in total boron
        r : float
            Correlation coefficient between PAR1 and PAR2
        
        Returns
        -------
        uncertainties : dict
            Dictionary mapping output variable names to their total uncertainties
        """

        # Ensure arrays for consistent handling
        PAR1 = np.atleast_1d(PAR1)
        PAR2 = np.atleast_1d(PAR2)
        ePAR1 = np.atleast_1d(ePAR1).astype(float)
        ePAR2 = np.atleast_1d(ePAR2).astype(float)
        r = np.atleast_1d(r).astype(float)
        
        # Convert error on pH to error on [H+] concentration
        # in case where first input variable is pH
        isH = (PAR1TYPE == 3)
        
        if np.any(isH):
            r = -r # Inverse sign of 'r' if PAR1 is pH
        
        # Same conversion for second variable
        isH = (PAR2TYPE == 3)
        if np.any(isH):
            r = -r # Inverse sign of 'r' if PAR2 is pH
        
        uncertainties = {}
        
        # For each output variable, compute total uncertainty
        for var in output_vars:
            # Initialize squared error
            sq_err = np.zeros_like(PAR1, dtype=float)
            
            # Contribution from PAR1
            if np.any(ePAR1 != 0):
                deriv_par1 = deriv[f"d_{var}__d_par1"]
                sq_err += (deriv_par1 * ePAR1)**2
            
            # Contribution from PAR2
            if np.any(ePAR2 != 0):
                deriv_par2 = deriv[f"d_{var}__d_par2"]
                sq_err += (deriv_par2 * ePAR2)**2
            
            # Covariance term (correlation between PAR1 and PAR2)
            if np.any(r != 0) and np.any(ePAR1 != 0) and np.any(ePAR2 != 0):
                covariance = r * ePAR1 * ePAR2
                sq_err += 2 * deriv_par1 * deriv_par2 * covariance
            
            # Contribution from salinity
            if np.any(eSAL != 0):
                deriv_sal = deriv[f"d_{var}__d_salinity"]
                sq_err += (deriv_sal * eSAL)**2
            
            # Contribution from temperature
            if np.any(eTEMP != 0):
                deriv_temp = deriv[f"d_{var}__d_temperature"]
                sq_err += (deriv_temp * eTEMP)**2
            
            # Contribution from silicate 
            if np.any(SI != 0) and np.any(eSI != 0):
                deriv_si = deriv[f"d_{var}__d_total_silicate"]
                sq_err += (deriv_si * eSI)**2
            
            # Contribution from phosphate
            if np.any(PO4 != 0) and np.any(ePO4 != 0):
                deriv_po4 = deriv[f"d_{var}__d_total_phosphate"]
                sq_err += (deriv_po4 * ePO4)**2
            
            # Contribution from dissociation constants
            pk_names = ["pk_CO2", "pk_carbonic_1", "pk_carbonic_2", 
                        "pk_borate", "pk_water", "pk_aragonite", "pk_calcite"]
            
            for i, pk_name in enumerate(pk_names):
                if epK[i] != 0:
                    deriv_pk = deriv[f"d_{var}__d_{pk_name}"]
                    sq_err += (deriv_pk * epK[i])**2
                
            # Contribution from total boron
            if eBt != 0:
                # Get total boron from results (need absolute error because eBt is given as fractional relative error)
                TB = deriv["total_borate"] # in umol/kg
                eBt_abs = eBt * TB  # absolute error
                deriv_bt = deriv[f"d_{var}__d_total_borate"]
                sq_err += (deriv_bt * eBt_abs)**2
            
            # Total uncertainty is sqrt of squared error
            uncertainties[var] = np.sqrt(sq_err)
        
        return uncertainties

@dataclass  
class ChemistryConstants:
        """Equilibrium constants for carbonate chemistry"""
        pk_CO2: float = 0.002
        pk_carbonic_1: float = 0.0075
        pk_carbonic_2: float = 0.015
        pk_borate: float = 0.01
        pk_water: float = 0.01
        pk_aragonite: float = 0.02
        pk_calcite: float = 0.02

@dataclass
class CalculationOptions:
        """Options for chemistry calculations"""
        pH_scale: int = 1 # pH scale used for any pH entries in `par1` and `par2`, see https://pyco2sys.readthedocs.io/en/latest/co2sys_nd/#arguments
        k_carbonic: int = 10 # set of equilibrium constants to use to model carbonic acid dissociation
        k_bisulfate: int = 1 # set of equilibrium constants to use to model bisulfate ion dissociation

@dataclass
class Measurements:
        """Environmental measurements"""
        salinity: Union[float, np.ndarray]
        temperature: Union[float, np.ndarray]
        pressure: Union[float, np.ndarray]
        total_silicate: float
        total_phosphate: float
        total_borate: float

@dataclass
class MeasurementErrors:
        """Measurement uncertainties"""
        salinity: float
        temperature: float

@dataclass
class CANYONData:
        """CANYON-B model outputs"""
        b_raw: Dict
        covariance: np.ndarray
        correlation: np.ndarray

def error_propagation(
            inpar: np.ndarray,
            carbonate_index: int, 
            parameter_1: Union[float, np.ndarray], 
            parameter_2: Union[float, np.ndarray],
            parameter_1_type: int, 
            parameter_2_type: int, 
            measurements: Measurements,
            errors: MeasurementErrors,
            canyon_data: CANYONData,
            constants: ChemistryConstants = None,
            options: CalculationOptions = None,
    ) -> tuple[Dict, Dict]:
        """
        Calculate carbonate system parameters and their uncertainties using error propagation.
    
        Parameters
        ----------
        inpar: np.ndarray
            Input parameter pairs for carbonate calculations
        carbonate_index : int
            Parameter index for selecting matrix elements
        parameter_1, parameter_2 : float or array-like
            First and second carbonate system parameter values
        parameter_1_type, parameter_2_type : int
            Type flags for parameters (PyCO2SYS convention)
        measurements : Measurements
            Environmental measurements (salinity, temperature, pressure, nutrients)
        errors : MeasurementErrors
            Uncertainties in salinity and temperature
        canyon_data : CANYONData
            CANYON-B raw results, covariance matrix, and correlation matrix
        constants : ChemistryConstants, optional
            Equilibrium constant uncertainties (default values if None)
        options : CalculationOptions, optional
            PyCO2SYS calculation options (default values if None)
        
        Returns
        -------
        deriv : dict
            Derivatives of carbonate system variables
        uncertainties : dict
            Propagated uncertainties for carbonate system variables
        """
            
        # Set defaults if not provided
        if constants is None:
            constants = ChemistryConstants()
        if options is None:
            options = CalculationOptions()
    
        # Unpack measurements 
        sal = measurements.salinity
        temp = measurements.temperature
        pres = measurements.pressure
        si = measurements.total_silicate
        po4 = measurements.total_phosphate
        bt = measurements.total_borate

        # Define measurement uncertainties
        uncertainty_from = {
            "par1": np.sqrt(canyon_data.covariance[inpar[carbonate_index, 0], inpar[carbonate_index, 0]]),
            "par2": np.sqrt(canyon_data.covariance[inpar[carbonate_index, 1], inpar[carbonate_index, 1]]),
            "temperature": errors.temperature,
            "salinity": errors.salinity,
            "total_silicate": canyon_data.b_raw['SiOH4']['eSiOH4'],
            "total_phosphate": canyon_data.b_raw['PO4']['ePO4'],
            "total_borate": bt,
            "pk_CO2": constants.pk_CO2,
            "pk_carbonic_1": constants.pk_carbonic_1,
            "pk_carbonic_2": constants.pk_carbonic_2,
            "pk_borate": constants.pk_borate,
            "pk_water": constants.pk_water,
            "pk_aragonite": constants.pk_aragonite,
            "pk_calcite": constants.pk_calcite
        }

        # Compute partial derivatives using PyCO2SYS
        deriv = pyco2.sys(
            par1=parameter_1,
            par2=parameter_2,
            par1_type=parameter_1_type,
            par2_type=parameter_2_type,
            salinity=sal,
            temperature=temp,
            pressure=pres,
            total_silicate=si,
            total_phosphate=po4,
            opt_pH_scale=options.pH_scale,
            opt_k_carbonic=options.k_carbonic,
            opt_k_bisulfate=options.k_bisulfate,
            grads_of=["pH", "pCO2", "alkalinity", "dic"],
            grads_wrt=["par1", "par2", "temperature", "salinity", "total_silicate", 
                    "total_phosphate", "total_borate", "pk_CO2", "pk_carbonic_1",
                    "pk_carbonic_2", "pk_borate", "pk_water", "pk_aragonite", 
                    "pk_calcite"]
        )
 
        # Prepare equilibrium constant uncertainties
        epK = [
        constants.pk_CO2, 
        constants.pk_carbonic_1, 
        constants.pk_carbonic_2, 
        constants.pk_borate, 
        constants.pk_water, 
        constants.pk_aragonite, 
        constants.pk_calcite
        ]

        # Propagate uncertainties
        uncertainties = calculate_uncertainties(
            deriv=deriv,
            output_vars=["pH", "pCO2", "alkalinity", "dic"],
            PAR1=parameter_1,
            PAR2=parameter_2,
            PAR1TYPE=parameter_1_type,
            PAR2TYPE=parameter_2_type,
            ePAR1=uncertainty_from["par1"],
            ePAR2=uncertainty_from["par2"],
            eSAL=uncertainty_from["salinity"],
            eTEMP=uncertainty_from["temperature"],
            SI=si,
            eSI=uncertainty_from["total_silicate"],
            PO4=po4,
            ePO4=uncertainty_from["total_phosphate"],
            eBt=uncertainty_from["total_borate"],
            epK=epK,
            r=canyon_data.correlation[inpar[carbonate_index,0],inpar[carbonate_index,1],]
        )
        
        return deriv, uncertainties