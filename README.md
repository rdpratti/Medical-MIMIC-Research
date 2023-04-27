# Medical-MIMIC-Research
 
 This project supports my work exploring the MIT MIMIC database.

 I am trying to answer the question on whether a model can be built to help support diagnostic decisions within the first few days of admittance to an ICU.

The project has four main notebooks:

1) Mimic_Note-EDA				Explores details about admission and note data by diagnosis

2) MIMIC-Scrub-Notes			Cleanup text data

3) MIMIC-Setup-For-Model-Work 	Creates, assigns labels, splits and balances files

4) Create-Test-Model			Fine-Tune Bio-ClinicalBert, Test and Measure Performance


The first three can easy run in a CPU environment. The last one was ported and executed on Kaggle using Hugging Face
