__author__ = 'sean.tokunaga@ifremer.fr'

import numpy as np

"""
Flags used to identify bad datasets. This is sort of an attempt to automatically detect an log anomalies in the
data structure. They should only really get used in the verify() function.

By convention, do not store anything that can't be read by pure python 
(otherwise, it's annoying for printing, json dumping etc.)
=> Stick to ints, floats, lists, str...etc. NO PANDAS CRAP and no NDARRAYS!
"""


# Errors caused by something being wrong in the dataset
class BadData(object):
    def __init__(self, description):
        self.description = description
        self.flag_id = None

class UnknownDataMode(BadData):
    def __init__(self, cycle_index):
        BadData.__init__(self, "unrecognized data mode")
        self.cycle_index = cycle_index.flatten().tolist()
        self.affected_cycle_n = len(self.cycle_index)
        self.flag_id = 0


class InappropriateAdjustedValues(BadData):
    def __init__(self, param, cycle_index, affected_measurement_n):
        BadData.__init__(self, "adjusted values inconsistent with data mode")
        self.param = param
        self.cycle_index = cycle_index.flatten().tolist()
        self.affected_cycle_n = len(self.cycle_index)
        self.affected_measurement_n = int(
            affected_measurement_n)  # estimated from # points according to data_mode selection

        self.flag_id = 1


class NegativePressure(BadData):
    def __init__(self, points, affected_cycle_n):
        BadData.__init__(self, "found negative pressures")
        self.points = points.tolist()
        self.affected_measurement_n = int(len(self.points))
        self.affected_cycle_n = int(affected_cycle_n)
        self.flag_id = 2


class MissingParameter(BadData):
    def __init__(self, param, affected_measurement_n, affected_cycle_n):
        BadData.__init__(self, "missing parameter")
        self.param = param
        self.affected_measurement_n = int(affected_measurement_n)  # Estimate by counting number of finite pres points.
        self.affected_cycle_n = int(affected_cycle_n)
        self.flag_id = 3


class BadQCCount(BadData):
    def __init__(self, param, qc_count, measurement_count):
        BadData.__init__(self, "qc count differs from measurement count")
        self.qc_name = param
        self.qc_count = int(qc_count)
        self.measurement_count = int(measurement_count)

        self.flag_id = 4


class QCCountsMismatch(BadData):
    """
    Trigger if we don't get the same number of psal/pres/temp qcs.
    CAUTION: This triggers a lot, and MAY be normal. It just means we don't have the same number of measurements,
    for the 3 parameters.
    """

    def __init__(self, affeted_measurements_n, affected_cycles_n):
        BadData.__init__(self, "qc counts don't match across params")
        self.affected_measurement_n = int(affeted_measurements_n)  # Estimate by counting number of finite pres points.
        self.affected_cycle_n = int(affected_cycles_n)

        self.flag_id = 5


class MeasurementCountsMismatch(BadData):
    def __init__(self, temp_n, psal_n, pres_n):
        BadData.__init__(self, "number of measurements don't match across params")
        self.temp_n = int(temp_n)
        self.pres_n = int(pres_n)
        self.psal_n = int(psal_n)

        self.flag_id = 6


class UnauthorisedNan(BadData):
    def __init__(self, param, points):
        BadData.__init__(self, "qc is nan for a measured value (or vice versa)")
        self.qc_name = param
        self.affected_measurement_n = int(len(points))  # Estimate by counting number of finite pres points.
        self.affected_cycle_n = int(len(np.unique(points[:, 0])))

        self.flag_id = 7


#############################################################################################
# FLAGS DURING VALUE SELECTION IN ARRAYS:


class LevelNotFound(BadData):
    def __init__(self, isas_15_alarm):
        BadData.__init__(self, "Level specified in alarm wasn't found in argo data.")
        self.isas_15_alarm = isas_15_alarm
        self.affected_measurement_n = 1  # 1 alarm point not found
        self.flag_id = 8


class PressureMismatch(BadData):
    def __init__(self, isas_15_alarm):
        BadData.__init__(self, "Pressure specified in alarm differs by more than 0.2 from argo data.")
        self.isas_15_alarm = isas_15_alarm
        self.affected_measurement_n = 1  # 1 alarm point not found
        self.flag_id = 9


class NonUniqueProfileSpecification(BadData):
    def __init__(self, isas_15_alarm):
        BadData.__init__(self, "Multiple profiles found for specified Cycle_Number + Direction.")
        self.isas_15_alarm = isas_15_alarm
        self.affected_measurement_n = 1  # 1 alarm point not found
        self.flag_id = 10


class ProfileNotFound(BadData):
    def __init__(self, isas_15_alarm):
        BadData.__init__(self, "Cycle_Number + Direction doesn't point to any profile in data.")
        self.isas_15_alarm = isas_15_alarm
        self.affected_measurement_n = 1  # 1 alarm point not found
        self.flag_id = 11


class DuplicateAlarmsFound(BadData):
    def __init__(self, isas_15_alarm):
        BadData.__init__(self, "Found multiple alarms for a single measurement.")
        self.isas_15_alarm = isas_15_alarm
        self.affected_measurement_n = 1  # 1 alarm point not found
        self.flag_id = 12

class InappropriateFloater(BadData):
    def __init__(self, affected_alarms_n):
        BadData.__init__(self, "Parameter missing (Temp or Psal).")
        self.affected_measurement_n = affected_alarms_n
        self.flag_id = 13