import subprocess
import traceback
from pymgipsim.Utilities.paths import results_path
from pymgipsim.Utilities import simulation_folder

from pymgipsim.Interface.parser import generate_parser_cli
from pymgipsim.InputGeneration.activity_settings import activity_args_to_scenario

from pymgipsim.generate_settings import generate_simulation_settings_main
from pymgipsim.generate_inputs import generate_inputs_main
from pymgipsim.generate_subjects import generate_virtual_subjects_main
from pymgipsim.generate_plots import generate_plots_main
from pymgipsim.generate_results import generate_results_main

def main():
    print(">>>>> Starting Simulation")
    try:
        args = generate_parser_cli().parse_args()
        subprocess.run(['python', 'initialization.py'])

        print(">>>>> Setting up results folder")
        _, _, _, results_folder_path = simulation_folder.create_simulation_results_folder(results_path)

        print(">>>>> Loading Scenario")
        settings_file = simulation_folder.load_settings_file(args, results_folder_path)

        args.controller_name = "GloopController"
        args.model_name = "T1DM.ExtHovorka"
        args.running_speed = 0.0
        args.plot_patient = 0
        args.breakfast_carb_range = [80, 120]
        args.am_snack_carb_range = [10, 20]
        args.lunch_carb_range = [80, 120]
        args.pm_snack_carb_range = [10, 20]
        args.dinner_carb_range = [80, 120]
        args.random_seed = 100

        activity_args_to_scenario(settings_file, args)

        if not args.scenario_name:
            print(">>>>> Generating Simulation Settings")
            settings_file = generate_simulation_settings_main(settings_file, args, results_folder_path)

            print(">>>>> Generating Virtual Cohort")
            settings_file = generate_virtual_subjects_main(settings_file, args, results_folder_path)

            print(">>>>> Generating Input Signals")
            settings_file = generate_inputs_main(settings_file, args, results_folder_path)

        print(">>>>> Generating Model Results")
        model, _ = generate_results_main(settings_file, vars(args), results_folder_path)

        if hasattr(model.model_solver.controller, "plot_results"):
            print("📊 Plotting results (controller based)")
            model.model_solver.controller.plot_results()

        print(">>>>> Generating Plots")
        _ = generate_plots_main(results_folder_path, args)

        print("✅ Simulation completed successfully!")

    except Exception as e:
        print("❌ An error occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
