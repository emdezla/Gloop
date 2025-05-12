# -*- coding: utf-8 -*-
"""
Created on Sat May 12

@author: Andy
"""
import numpy as np
import os, json, tqdm, glob
from pymgipsim.Utilities.paths import default_settings_path, results_path, simulator_path
from pymgipsim.Utilities.units_conversions_constants import DEFAULT_RANDOM_SEED
from pymgipsim.VirtualPatient.VirtualPatient import VirtualCohort
from pymgipsim.Utilities.Scenario import scenario, save_scenario
from pymgipsim.Utilities import simulation_folder
from pymgipsim.VirtualPatient.Models import T1DM, Multiscale
from pymgipsim.VirtualPatient.parser import generate_virtual_subjects_parser
from pymgipsim.Settings.parser import generate_settings_parser

from pymgipsim.generate_settings import generate_simulation_settings_main
from pymgipsim.Utilities.simulation_folder import get_most_recent_folder_in_directory
import argparse, json, pprint
from dataclasses import asdict

np.random.seed(DEFAULT_RANDOM_SEED)

def list_model_patients(model_name: str):
    path = os.path.join(simulator_path, "VirtualPatient", "Models", model_name.replace(".", "/"), "Patients", "*.json")
    files = []
    for filename in glob.iglob(path):
        files.append(os.path.split(filename)[-1])

    return files


def generate_patient_names(scenario_instance: scenario, args: argparse.Namespace):
    try:
        if scenario_instance.patient.files is None:
            all_files = list_model_patients(scenario_instance.patient.model.name)
            np.random.seed(DEFAULT_RANDOM_SEED)
            patient_indeces = np.random.choice(len(all_files), size=(scenario_instance.patient.number_of_subjects,), replace=False)
            cohort_paths = [all_files[i] for i in patient_indeces]
            scenario_instance.patient.files = cohort_paths
        else:
            scenario_instance.patient.number_of_subjects = len(scenario_instance.patient.files)
            args.number_of_subjects = scenario_instance.patient.number_of_subjects
        scenario_instance.patient.files = [file if ".json" in file else file + ".json" for file in scenario_instance.patient.files]
        print("")
    except:
        scenario_instance.patient.files = None


def generate_virtual_subjects_main(scenario_instance: scenario, args: argparse.Namespace, results_folder_path: str):

    if not args.no_print:
        print(f">>>>> Generating Virtual Cohort")

    results_folder_path = get_most_recent_folder_in_directory(results_path)

    scenario_instance.patient.model.name = args.model_name
    scenario_instance.patient.files = args.patient_names
    scenario_instance.patient.number_of_subjects = args.number_of_subjects

    scenario_instance.patient.demographic_info.renal_function_category = args.renal_function_category
    scenario_instance.patient.demographic_info.body_weight_range = args.body_weight_range

    generate_patient_names(scenario_instance, args)

    """ Load/generate demographic and multiscale info """
    scenario_instance = VirtualCohort.generate_demographic_info(scenario_instance)

    """ Generate virtual parameters """

    match scenario_instance.patient.model.name:
        case T1DM.IVP.Model.name:
            parameters = T1DM.IVP.Parameters.generate(scenario_instance)
        case T1DM.ExtHovorka.Model.name:
            parameters = T1DM.ExtHovorka.Parameters.generate(scenario_instance)

    scenario_instance.patient.model.parameters = parameters.tolist()

    parameters = Multiscale.BodyWeight.Parameters.generate(scenario_instance)
    scenario_instance.patient.mscale.parameters = parameters.tolist()

    save_scenario(os.path.join(results_folder_path, "simulation_settings.json"), asdict(scenario_instance))

    return scenario_instance


if __name__ == '__main__':


    with open(os.path.join(default_settings_path, "scenario_default.json"), "r") as f: #
        default_scenario = scenario(**json.load(f))
    f.close()

    """ Define Results Path """
    _, _, _, results_folder_path = simulation_folder.create_simulation_results_folder(results_path)


    """ Parse Arguments  """
    parser = generate_virtual_subjects_parser(parent_parser=[generate_settings_parser(add_help = False)])
    args = parser.parse_args()

    settings_file = generate_simulation_settings_main(scenario_instance=default_scenario, args=args, results_folder_path = results_folder_path)

    settings_file = generate_virtual_subjects_main(scenario_instance = settings_file, args=args, results_folder_path = results_folder_path)

    if args.verbose:
        pprint.PrettyPrinter(indent=2).pprint(settings_file)
