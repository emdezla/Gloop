import os
from pymgipsim.Utilities.Scenario import scenario, demographic_info
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from .Models import T1DM, Multiscale
from pymgipsim.VirtualPatient.Models.Multiscale.BodyWeight.CONSTANTS import NOMINAL_EI0, NOMINAL_EE0, NOMINAL_UGE0
from pymgipsim.Utilities.paths import models_path
from pymgipsim.ModelSolver import multiscale, singlescale
import numpy as np
import json

class VirtualCohort:
    """ Abstract virtual patient class.
    """
    singlescale_model = None
    multiscale_model = None

    def __init__(self, scenario_instance: scenario):

        # Force GloopController to enable closed-loop regulation
        scenario_instance.controller.name = 'GloopController'

        # Increase basal insulin delivery manually for all subjects
        n_subjects = scenario_instance.patient.number_of_subjects
        scenario_instance.patient.demographic_info.basal = [2.5] * n_subjects

        match scenario_instance.patient.model.name:
            case T1DM.ExtHovorka.Model.name:
                self.singlescale_model = T1DM.ExtHovorka.Model.from_scenario(scenario_instance=scenario_instance)
            case T1DM.IVP.Model.name:
                self.singlescale_model = T1DM.IVP.Model.from_scenario(scenario_instance=scenario_instance)
            case _:
                raise Exception(f"Unknown model name {scenario_instance.patient.model.name}")

        self.singlescale_model_solver = singlescale.SingleScaleSolver(scenario_instance, self.singlescale_model)
        self.model_solver = self.singlescale_model_solver

        if scenario_instance.settings.simulator_name == 'MultiScaleSolver':
            self.multiscale_model = Multiscale.BodyWeight.Model.from_scenario_multiscale(scenario_instance=scenario_instance)
            self.multiscale_model_solver = multiscale.MultiScaleSolver(scenario_instance, self.singlescale_model, self.multiscale_model)
            self.model_solver = self.multiscale_model_solver

    @property
    def glucose(self):
        return self.singlescale_model_solver.model.states.as_array[:, self.model_solver.model.glucose_state, :]

    @staticmethod
    def generate_demographic_info(scenario_instance: scenario):

        if not scenario_instance.patient.demographic_info:
            scenario_instance.patient.demographic_info = demographic_info()

        try:
            body_weight, egfr, basal, height, total_daily_basal, carb_insulin_ratio, resting_heart_rate, correction_bolus, hba1c, waist_size, baseline_daily_energy_intake, baseline_daily_energy_expenditure, baseline_daily_urinary_glucose_excretion = [], [], [], [], [], [], [], [], [], [], [], [], []
            path = os.path.join(models_path, scenario_instance.patient.model.name.replace(".", os.sep), "Patients")
            for name in scenario_instance.patient.files:
                abs_path = os.path.join(path, name)
                with open(abs_path) as f:
                    params_dict = json.load(f)
                dem = demographic_info(**params_dict["demographic_info"])
                baseline_daily_energy_intake.append(dem.baseline_daily_energy_intake)
                baseline_daily_energy_expenditure.append(dem.baseline_daily_energy_expenditure)
                baseline_daily_urinary_glucose_excretion.append(dem.baseline_daily_urinary_glucose_excretion)
                body_weight.append(dem.body_weight)
                egfr.append(dem.egfr)
                basal.append(dem.basal)
                height.append(dem.height)
                total_daily_basal.append(dem.total_daily_basal)
                carb_insulin_ratio.append(dem.carb_insulin_ratio)
                resting_heart_rate.append(dem.resting_heart_rate)
                correction_bolus.append(dem.correction_bolus)
                hba1c.append(dem.HbA1c)
                waist_size.append(dem.waist_size)
        except:
            n_subjects = scenario_instance.patient.number_of_subjects
            body_weight = np.random.uniform(low=scenario_instance.patient.demographic_info.body_weight_range[0],
                                            high=scenario_instance.patient.demographic_info.body_weight_range[1],
                                            size=n_subjects).tolist()
            basal = n_subjects * [1.7]
            height = n_subjects * [175]
            total_daily_basal = n_subjects * [24]
            carb_insulin_ratio = n_subjects * [12.239]
            resting_heart_rate = n_subjects * [70.0]
            correction_bolus = n_subjects * [25.0]
            hba1c = n_subjects * [0.0]
            waist_size = n_subjects * [0.0]
            baseline_daily_energy_intake = n_subjects * [NOMINAL_EI0]
            baseline_daily_energy_expenditure = n_subjects * [NOMINAL_EE0]
            baseline_daily_urinary_glucose_excretion = n_subjects * [NOMINAL_UGE0]
            scenario_instance.patient.files = None

            match scenario_instance.patient.demographic_info.renal_function_category:
                case 1:
                    low, high = 90, 100
                case 2:
                    low, high = 60, 89
                case 3:
                    low, high = 30, 59
                case 4:
                    low, high = 15, 29
                case 5:
                    low, high = 5, 15

            egfr = np.random.uniform(low=UnitConversion.metric.milli_to_base(low), high=UnitConversion.metric.milli_to_base(high), size=n_subjects).tolist()

        scenario_instance.patient.demographic_info.body_weight = body_weight
        scenario_instance.patient.demographic_info.egfr = egfr
        scenario_instance.patient.demographic_info.basal = basal
        scenario_instance.patient.demographic_info.height = height
        scenario_instance.patient.demographic_info.total_daily_basal = total_daily_basal
        scenario_instance.patient.demographic_info.carb_insulin_ratio = carb_insulin_ratio
        scenario_instance.patient.demographic_info.resting_heart_rate = resting_heart_rate
        scenario_instance.patient.demographic_info.correction_bolus = correction_bolus
        scenario_instance.patient.demographic_info.HbA1c = hba1c
        scenario_instance.patient.demographic_info.waist_size = waist_size
        scenario_instance.patient.demographic_info.baseline_daily_energy_intake = baseline_daily_energy_intake
        scenario_instance.patient.demographic_info.baseline_daily_energy_expenditure = baseline_daily_energy_expenditure
        scenario_instance.patient.demographic_info.baseline_daily_urinary_glucose_excretion = baseline_daily_urinary_glucose_excretion

        return scenario_instance