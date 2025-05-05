import pandas as pd
import matplotlib.pyplot as plt


def generate_imgs_for_training(data, index, indent=50, pid="", dirname="img_data", verbose=False, figsize=(3,3)):

    # Ensure that 'time' column is a datetime object
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')  # matches the full datetime format
    data['time_str'] = data['time'].dt.strftime('%H:%M:%S')  # format to only output hour, minute, and second
    
    # Extract the patch
    patch = data[index-indent:index].copy()

    # Calculate time since the current index
    current_time = patch.iloc[-1]['time']
    patch['time_since'] = (patch['time'] - current_time).dt.total_seconds() / 60
    
    # Plot the data with the specified figure size
    for metric, ylabel, filename in [
        ('glu_raw', 'Glucose Raw', 'glu'),  # glucose
        ('hr', 'Heart Rate Raw', 'hr'),     # heart rate
        ('iob', 'Insulin On Body', 'iob'),  # insulin on body
        ('basal', 'Basal', 'basal'),        # basal
        ('bolus', 'Bolus', 'bolus')         # bolus
        ]:

        # Create the plot
        plt.figure(figsize=figsize)
        plt.plot(patch['time_since'], patch[metric])
        plt.xlabel('Time (minutes)')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} Over Time')
        plt.tight_layout()
        plt.savefig(f"./{dirname}/{pid}_{index}_{filename}.png", dpi=70)

        # add urls to df
        data.at[index, f"{filename}_url"] = f"./{dirname}/{pid}_{index}_{filename}.png"
        
        if verbose: plt.show()
        else: plt.close()

    # data.at[index,"hr_url"] = f"./img_data/{pid}_{index}_hr.png"
    # data.at[index,"iob_url"] = f"./img_data/{pid}_{index}_iob.png"
    # data.at[index,"basal_url"] = f"./img_data/basal_{index}.png"
    # data.at[index,"bolus_url"] = f"./img_data/bolus_{index}.png"

    # Sum the data for the patch for basal and bolus for index-indent:index to ensure nothing is missed
    #data.at[index, 'basal_interval_sum'] = patch['basal'].sum()
    #data.at[index, 'bolus_interval_sum'] = patch['bolus'].sum()

    return data


# QUESTIONS
# what text to associate?
# what data to include?
# interval length? randomize length?
# fixed ylim?
# discrete tokenization 

#------

# format for Llama Factory
# def format_as_chat(img_paths, text, output=""):

#   return {
#     "messages": [
#       {
#         "content": f"<image><image><image>{text}",
#         "role": "user"
#       },
#       {
#         "content": output,
#         "role": "assistant"
#       }
#     ],
#     "images": [
#       img_paths,
#     ]
#   }

# # Text description
# text = "You predict basal and bolus"

# # Create json format for Llama factory
# data_json = [
#     format_as_chat(
#       [row["glu_url"],row["hr_url"],row["iob_url"]],
#       text = text,
#       output = f"{int(row['basal_binned'])} {int(row['bolus_binned'])}",
#     ) for i, row in df.dropna().iterrows()]


# # Store JSON data in a file
# with open(f"{pid}-train.json", 'w') as file:
#     json.dump(data_json, file, indent=4)


    