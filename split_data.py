# Split Data
import splitfolders

input_folder = 'garbage_classification' 
output_folder = 'dataset' 

splitfolders.ratio(input_folder,
                   output=output_folder, 
                   seed=42, 
                   ratio=(.8, .1, .1), 
                   group_prefix=None, 
                   move=False)